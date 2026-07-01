//! # AppRuntime — the L4 runtime composition root and live app state.
//!
//! DESIGN §19 pins the `app/` layer as "Tauri commands, wiring (**thin**)". This
//! module is where the *wiring* and the *live state* live, so the Tauri command
//! handlers in `lib.rs` stay thin: they translate DTOs and delegate here.
//!
//! [`AppRuntime`] owns the real L1/L2 adapters ([`SqliteStore`], [`SocketSidecar`]),
//! the single M1 watched folder + config, and the two pieces of genuinely stateful
//! runtime behavior that would otherwise leak across several command/protocol
//! handlers:
//!
//! 1. **The indexing run.** [`IndexRun`] plus the concurrency guard in
//!    [`AppRuntime::start_index`] and the phase state machine in
//!    [`AppRuntime::index_status`] (see [`derive_phase`]). Two commands read/write
//!    one `current_run`; the "reject a second overlapping run" and "derive
//!    Idle/Scanning/Indexing from progress" rules live in one place.
//! 2. **The Sidecar child process.** [`SidecarChild`] spawns the Python sidecar and
//!    kills + waits for it on drop — one owner, one lifecycle, dropped with the
//!    runtime.
//!
//! It also owns the path-traversal-safe `lume://thumb/<id>` resolution
//! ([`resolve_thumbnail`]) and query normalization, keeping the protocol handler
//! and `search` command free of business logic.
//!
//! Deletion test (DESIGN §19 discipline): delete this module and the indexing-run
//! guard/phase logic, the sidecar-child kill-on-drop, adapter construction, and
//! the thumbnail path guard all reappear scattered across `run`, the protocol
//! closure, and three command handlers.

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use lume_core::rank::{rank_tiles, Tile, TileMeta};
use lume_core::{Config, FileId, MediaKind, SearchFilters, Sidecar, VectorStore};
use lume_index::{Indexer, Progress};
use lume_ipc::SocketSidecar;
use lume_store::SqliteStore;

const SIDECAR_SOCKET_NAME: &str = "sidecar.sock";

/// The composition root's live state: the real L1/L2 adapters, the one M1 watched
/// folder, whatever indexing run is in flight, and the owned sidecar child. This
/// is the single value the Tauri layer manages as state.
pub struct AppRuntime {
    store: Arc<SqliteStore>,
    sidecar: Arc<SocketSidecar>,
    config: Config,
    watch_folder: PathBuf,
    thumbnails_dir: PathBuf,
    current_run: Mutex<Option<IndexRun>>,
    /// Held so the Python sidecar is killed + waited when the runtime drops
    /// (its `Drop` does the work). Never read directly.
    _sidecar_child: SidecarChild,
}

/// One in-flight (or most-recently-finished) `Indexer::run`, so `start_index` can
/// guard against a second concurrent run and `index_status` can report real
/// progress (M1 Slice 3's [`Progress`] handle).
struct IndexRun {
    handle: JoinHandle<()>,
    progress: Arc<Progress>,
}

/// Owns the spawned Python sidecar child process. The child is killed and waited
/// on drop, in this **one** place (DESIGN §19: platform/process lifecycle stays
/// local, never sprinkled).
struct SidecarChild(Mutex<Option<Child>>);

/// Coarse phase the runtime derives from the live indexing run. The command seam
/// maps this to the webview-facing `IndexPhase` DTO; the runtime never speaks the
/// wire enum. M1 only ever produces these three; `Paused`/`Error` arrive with M2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPhase {
    Idle,
    Scanning,
    Indexing,
}

/// A snapshot of the current indexing run for the command seam to translate.
#[derive(Debug, Clone, Copy)]
pub struct RunStatus {
    pub phase: RunPhase,
    pub done: u64,
    pub total: u64,
}

impl AppRuntime {
    /// Construct every adapter and live-state field: create `~/.lume` +
    /// `thumbnails`, open the store, dial the sidecar socket, spawn the Python
    /// sidecar child, and resolve the M1 watched folder. This is the composition
    /// root — the one place adapter wiring lives.
    pub fn bootstrap() -> Self {
        let data_dir = lume_data_dir();
        ensure_private_dir(&data_dir).expect("create private ~/.lume");
        let thumbnails_dir = data_dir.join("thumbnails");
        ensure_private_dir(&thumbnails_dir).expect("create private ~/.lume/thumbnails");
        let socket_path = sidecar_socket_path(&data_dir);

        let config = Config::default();
        let store =
            Arc::new(SqliteStore::open(data_dir.join("lume.sqlite3")).expect("open SqliteStore"));
        let sidecar = Arc::new(SocketSidecar::new(&socket_path, config.thumbnails.grid_px));

        AppRuntime {
            store,
            sidecar,
            watch_folder: watch_folder_from_env(),
            config,
            thumbnails_dir,
            current_run: Mutex::new(None),
            _sidecar_child: SidecarChild(Mutex::new(spawn_sidecar(&socket_path))),
        }
    }

    /// Semantic search → ranked Tiles (DESIGN §12). Normalizes the query, then
    /// over-fetches Units and **delegates** collapse → floor → cap → cliff to
    /// [`rank_tiles`]; owns no cutoff logic. Returns core [`Tile`]s — the command
    /// seam turns them into `SearchHit` DTOs.
    pub fn search(&self, raw_query: &str, filters: SearchFilters) -> Result<Vec<Tile>, String> {
        let query = normalize_query(raw_query);
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let query_emb = self.sidecar.embed_text(&query).map_err(|e| e.to_string())?;
        let results = &self.config.results;
        // Over-fetch Units so dedup still yields a full grid (DESIGN §12).
        let k = results.tile_cap * results.unit_oversample;

        let hits = self
            .store
            .knn(&query_emb, k, &filters)
            .map_err(|e| e.to_string())?;

        // M1 indexes only images (BUILD.md scope): every Item is one image Unit,
        // its thumbnail identity is its own FileId. M3 swaps this resolver for
        // real stored MediaKind + poster identity with no change to the pipeline.
        Ok(rank_tiles(hits, results, m1_image_meta).tiles)
    }

    /// Kick off (or resume) indexing of the watched folder. Errors if a run is
    /// already in progress rather than starting a second, overlapping one — the
    /// concurrency guard the runtime owns so no command has to.
    pub fn start_index(&self) -> Result<(), String> {
        let mut current = self.current_run.lock().expect("current_run lock poisoned");
        if let Some(run) = current.as_ref() {
            if !run.handle.is_finished() {
                return Err("indexing already in progress".into());
            }
        }

        let indexer = Indexer::new(
            self.watch_folder.clone(),
            self.config.batch_size,
            self.thumbnails_dir.clone(),
            Arc::clone(&self.store),
            Arc::clone(&self.sidecar) as Arc<dyn Sidecar + Send + Sync>,
        );
        let progress = indexer.progress();
        let handle = std::thread::spawn(move || {
            if let Err(err) = indexer.run() {
                tracing::error!(%err, "indexing run failed");
            }
        });

        *current = Some(IndexRun { handle, progress });
        Ok(())
    }

    /// Poll current indexing progress, deriving the coarse [`RunPhase`] from the
    /// run's `Progress` snapshot and thread liveness (see [`derive_phase`]).
    pub fn index_status(&self) -> RunStatus {
        let current = self.current_run.lock().expect("current_run lock poisoned");
        let Some(run) = current.as_ref() else {
            return RunStatus {
                phase: RunPhase::Idle,
                done: 0,
                total: 0,
            };
        };

        let (done, total) = run.progress.snapshot();
        RunStatus {
            phase: derive_phase(run.handle.is_finished(), total),
            done,
            total,
        }
    }

    /// Serve `lume://thumb/<file_id>` from the runtime's thumbnails dir. The
    /// path-traversal guard lives in [`resolve_thumbnail`].
    pub fn thumbnail_response(
        &self,
        request: &tauri::http::Request<Vec<u8>>,
    ) -> tauri::http::Response<Vec<u8>> {
        resolve_thumbnail(&self.thumbnails_dir, request)
    }
}

impl Drop for SidecarChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.lock().expect("sidecar child lock poisoned").take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// The indexing-run phase state machine. Pure so the transitions are testable
/// without booting Tauri, the store, or a real thread.
fn derive_phase(finished: bool, total: u64) -> RunPhase {
    if finished {
        RunPhase::Idle
    } else if total == 0 {
        RunPhase::Scanning
    } else {
        RunPhase::Indexing
    }
}

/// M1's Item-metadata resolver: everything is an image whose thumbnail is itself.
fn m1_image_meta(file: FileId) -> TileMeta {
    TileMeta {
        kind: MediaKind::Image,
        thumb_id: file,
    }
}

fn normalize_query(query: &str) -> String {
    query.trim().to_lowercase()
}

/// Resolve `lume://thumb/<file_id>` to `<thumbnails_dir>/<file_id>.jpg` (DESIGN
/// §14 — never HTTP, never a raw filesystem path from the webview). A non-integer
/// id is rejected: parsing the id to an `i64` doubles as the path-traversal guard.
fn resolve_thumbnail(
    thumbnails_dir: &Path,
    request: &tauri::http::Request<Vec<u8>>,
) -> tauri::http::Response<Vec<u8>> {
    let file_id = request.uri().path().trim_start_matches('/').parse::<i64>();

    let Ok(file_id) = file_id else {
        return tauri::http::Response::builder()
            .status(400)
            .body(Vec::new())
            .expect("static response is well-formed");
    };

    match std::fs::read(thumbnails_dir.join(format!("{file_id}.jpg"))) {
        Ok(bytes) => tauri::http::Response::builder()
            .status(200)
            .header("Content-Type", "image/jpeg")
            .body(bytes)
            .expect("static response is well-formed"),
        Err(_) => tauri::http::Response::builder()
            .status(404)
            .body(Vec::new())
            .expect("static response is well-formed"),
    }
}

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME must be set on macOS"))
}

fn lume_data_dir() -> PathBuf {
    home_dir().join(".lume")
}

fn ensure_private_dir(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

fn sidecar_socket_path(data_dir: &Path) -> PathBuf {
    data_dir.join(SIDECAR_SOCKET_NAME)
}

/// M1's stand-in for Settings-driven folder configuration (M6): one watched
/// folder from `LUME_WATCH_FOLDER`, defaulting to `~/Pictures`.
fn watch_folder_from_env() -> PathBuf {
    match std::env::var_os("LUME_WATCH_FOLDER") {
        Some(path) => PathBuf::from(path),
        None => home_dir().join("Pictures"),
    }
}

fn spawn_sidecar(socket_path: &Path) -> Option<Child> {
    if std::env::var_os("LUME_DISABLE_SIDECAR").is_some() {
        return None;
    }

    let sidecar_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri has a workspace parent")
        .join("sidecar");
    let mut args = vec![
        "run".to_string(),
        "python".to_string(),
        "-m".to_string(),
        "lume_sidecar.server".to_string(),
        "--socket".to_string(),
        socket_path.to_string_lossy().into_owned(),
    ];
    if std::env::var_os("LUME_SIDECAR_FAKE_EMBEDDER").is_some() {
        args.push("--fake".to_string());
    }
    let child = Command::new("uv")
        .args(args)
        .current_dir(sidecar_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    match child {
        Ok(child) => Some(child),
        Err(err) => {
            eprintln!("Sidecar spawn skipped: {err}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        derive_phase, ensure_private_dir, normalize_query, resolve_thumbnail, sidecar_socket_path,
        RunPhase,
    };
    use std::os::unix::fs::PermissionsExt;

    fn request(uri: &str) -> tauri::http::Request<Vec<u8>> {
        tauri::http::Request::builder()
            .uri(uri)
            .body(Vec::new())
            .unwrap()
    }

    #[test]
    fn serves_the_thumbnail_bytes_for_a_known_id() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("42.jpg"), b"jpeg-bytes").unwrap();

        let resp = resolve_thumbnail(dir.path(), &request("lume://thumb/42"));

        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("Content-Type").unwrap(), "image/jpeg");
        assert_eq!(resp.body(), b"jpeg-bytes");
    }

    #[test]
    fn returns_404_for_a_missing_thumbnail() {
        let dir = tempfile::tempdir().unwrap();
        let resp = resolve_thumbnail(dir.path(), &request("lume://thumb/999"));
        assert_eq!(resp.status(), 404);
    }

    #[test]
    fn rejects_non_integer_ids_as_a_path_traversal_guard() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("1.jpg"), b"jpeg-bytes").unwrap();

        for uri in [
            "lume://thumb/../../etc/passwd",
            "lume://thumb/1.jpg",
            "lume://thumb/abc",
            "lume://thumb/",
        ] {
            let resp = resolve_thumbnail(dir.path(), &request(uri));
            assert_eq!(resp.status(), 400, "expected 400 for {uri}");
        }
    }

    #[test]
    fn normalizes_query_before_embedding() {
        assert_eq!(
            normalize_query("  Girl Riding A Bicycle  "),
            "girl riding a bicycle"
        );
        assert_eq!(normalize_query("\n\tSUNSET\t\n"), "sunset");
    }

    #[test]
    fn sidecar_socket_lives_inside_private_lume_data_dir() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join(".lume");

        ensure_private_dir(&data_dir).unwrap();
        let socket = sidecar_socket_path(&data_dir);

        assert!(socket.starts_with(&data_dir));
        assert_ne!(socket, std::path::PathBuf::from("/tmp/lume-sidecar.sock"));
        assert_eq!(
            std::fs::metadata(&data_dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }

    #[test]
    fn index_run_phase_transitions() {
        // Before any progress is recorded a live run is still Scanning.
        assert_eq!(derive_phase(false, 0), RunPhase::Scanning);
        // Once the walk has set a total, a live run is Indexing.
        assert_eq!(derive_phase(false, 12), RunPhase::Indexing);
        // A finished thread is Idle regardless of the last counters.
        assert_eq!(derive_phase(true, 0), RunPhase::Idle);
        assert_eq!(derive_phase(true, 12), RunPhase::Idle);
    }
}
