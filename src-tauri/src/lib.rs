//! # lume-app (L4 application)
//!
//! The Tauri shell and the **typed command surface** the UI calls: the
//! composition root that constructs the real [`VectorStore`], [`Sidecar`],
//! and [`Indexer`] adapters and wires them into the commands below. The UI
//! never sees any of that — it only invokes the commands here (DESIGN §19:
//! "Tauri commands = typed API; UI holds zero business logic").
//!
//! The DTOs are webview-facing (camelCase JSON) and mirror `src/lib/commands.ts`.
//! Keep the two in lockstep.
//!
//! M1 has exactly one watched folder, read from `LUME_WATCH_FOLDER`
//! (`~/Pictures` default) — an explicit stand-in for real Settings-driven
//! folder configuration, which is M6.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use lume_core::{Config, MediaKind, ScoredHit, Sidecar, VectorStore};
use lume_index::{Indexer, Progress};
use lume_ipc::SocketSidecar;
use lume_store::SqliteStore;
use serde::{Deserialize, Serialize};
use tauri::Manager;

const SIDECAR_SOCKET_PATH: &str = "/tmp/lume-sidecar.sock";

struct SidecarChild(Mutex<Option<Child>>);

/// One result Tile (DESIGN §12). Deduplicated from Units before it crosses here.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub file_id: i64,
    /// `lume://` URL for the stored grid thumbnail / video poster (DESIGN §14).
    pub thumb_url: String,
    pub kind: HitKind,
    pub score: f32,
    /// Matched video-frame timestamps for scrubber markers (DESIGN §7).
    pub matched_timestamps: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HitKind {
    Image,
    Video,
}

/// Structured filters combined with the semantic query (DESIGN §12).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchFilters {
    pub kind: Option<HitKindFilter>,
    pub captured_after: Option<i64>,
    pub captured_before: Option<i64>,
    pub folder: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HitKindFilter {
    Image,
    Video,
}

/// Coarse indexing lifecycle for the menu bar + onboarding (DESIGN §11, §18).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexStatus {
    pub phase: IndexPhase,
    pub done: u64,
    pub total: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexPhase {
    Idle,
    Scanning,
    Indexing,
    Paused,
    Error,
}

/// One in-flight (or most-recently-finished) `Indexer::run`, so `start_index`
/// can guard against a second concurrent run and `index_status` can report
/// real progress (M1 Slice 3's [`Progress`] handle).
struct IndexRun {
    handle: JoinHandle<()>,
    progress: Arc<Progress>,
}

/// The composition root's live state: the real L1/L2 adapters, the one M1
/// watched folder, and whatever indexing run is currently in flight.
struct Engine {
    store: Arc<SqliteStore>,
    sidecar: Arc<SocketSidecar>,
    config: Config,
    watch_folder: PathBuf,
    thumbnails_dir: PathBuf,
    current_run: Mutex<Option<IndexRun>>,
}

/// Semantic search → ranked Tiles (DESIGN §12). No adaptive floor/cliff yet
/// (M4) — M1 only needs "ranked, capped."
#[tauri::command]
fn search(
    engine: tauri::State<Engine>,
    query: String,
    filters: Option<SearchFilters>,
) -> Result<Vec<SearchHit>, String> {
    let query_emb = engine
        .sidecar
        .embed_text(&query)
        .map_err(|e| e.to_string())?;
    let core_filters = to_core_filters(filters);
    let k = engine.config.results.tile_cap * engine.config.results.unit_oversample;

    let hits = engine
        .store
        .knn(&query_emb, k, &core_filters)
        .map_err(|e| e.to_string())?;

    Ok(hits
        .into_iter()
        .take(engine.config.results.tile_cap)
        .map(to_search_hit)
        .collect())
}

/// Kick off (or resume) indexing of the watched folder. Errors if a run is
/// already in progress rather than starting a second, overlapping one.
#[tauri::command]
fn start_index(engine: tauri::State<Engine>) -> Result<(), String> {
    let mut current = engine
        .current_run
        .lock()
        .expect("current_run lock poisoned");
    if let Some(run) = current.as_ref() {
        if !run.handle.is_finished() {
            return Err("indexing already in progress".into());
        }
    }

    let indexer = Indexer::new(
        engine.watch_folder.clone(),
        engine.config.batch_size,
        engine.thumbnails_dir.clone(),
        Arc::clone(&engine.store),
        Arc::clone(&engine.sidecar) as Arc<dyn Sidecar + Send + Sync>,
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

/// Poll current indexing progress.
#[tauri::command]
fn index_status(engine: tauri::State<Engine>) -> IndexStatus {
    let current = engine
        .current_run
        .lock()
        .expect("current_run lock poisoned");
    let Some(run) = current.as_ref() else {
        return IndexStatus {
            phase: IndexPhase::Idle,
            done: 0,
            total: 0,
        };
    };

    let (done, total) = run.progress.snapshot();
    let phase = if run.handle.is_finished() {
        IndexPhase::Idle
    } else if total == 0 {
        IndexPhase::Scanning
    } else {
        IndexPhase::Indexing
    };

    IndexStatus { phase, done, total }
}

fn to_core_filters(filters: Option<SearchFilters>) -> lume_core::SearchFilters {
    let filters = filters.unwrap_or_default();
    lume_core::SearchFilters {
        kind: filters.kind.map(|k| match k {
            HitKindFilter::Image => MediaKind::Image,
            HitKindFilter::Video => MediaKind::Video,
        }),
        captured_after: filters.captured_after,
        captured_before: filters.captured_before,
        folder: filters.folder.map(PathBuf::from),
    }
}

fn to_search_hit(hit: ScoredHit) -> SearchHit {
    SearchHit {
        file_id: hit.file,
        thumb_url: format!("lume://thumb/{}", hit.file),
        // M1's walker only ever indexes images (BUILD.md scope) — every Unit
        // is 1:1 with its Item, so this is correct today. Video (M3) is what
        // makes `kind` genuinely need to come from the file's stored
        // MediaKind instead.
        kind: HitKind::Image,
        score: hit.score,
        matched_timestamps: hit.frame_ts.map(|t| vec![t]).unwrap_or_default(),
    }
}

/// Serve `lume://thumb/<file_id>` from `~/.lume/thumbnails/<file_id>.jpg`
/// (DESIGN §14 — never HTTP, never a raw filesystem path from the webview).
/// A non-integer id is rejected: this doubles as the path-traversal guard.
fn thumb_protocol_response(
    thumbnails_dir: &std::path::Path,
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

/// Build and run the Tauri application. Called by `main.rs`.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .register_uri_scheme_protocol("lume", |ctx, request| {
            let engine = ctx.app_handle().state::<Engine>();
            thumb_protocol_response(&engine.thumbnails_dir, &request)
        })
        .setup(|app| {
            app.manage(SidecarChild(Mutex::new(spawn_sidecar())));

            let data_dir = lume_data_dir();
            std::fs::create_dir_all(&data_dir).expect("create ~/.lume");
            let thumbnails_dir = data_dir.join("thumbnails");
            std::fs::create_dir_all(&thumbnails_dir).expect("create ~/.lume/thumbnails");

            let config = Config::default();
            let store = Arc::new(
                SqliteStore::open(data_dir.join("lume.sqlite3")).expect("open SqliteStore"),
            );
            let sidecar = Arc::new(SocketSidecar::new(
                SIDECAR_SOCKET_PATH,
                config.thumbnails.grid_px,
            ));

            app.manage(Engine {
                store,
                sidecar,
                watch_folder: watch_folder_from_env(),
                config,
                thumbnails_dir,
                current_run: Mutex::new(None),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![search, start_index, index_status])
        .run(tauri::generate_context!())
        .expect("error while running Lume");
}

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME must be set on macOS"))
}

fn lume_data_dir() -> PathBuf {
    home_dir().join(".lume")
}

/// M1's stand-in for Settings-driven folder configuration (M6): one watched
/// folder from `LUME_WATCH_FOLDER`, defaulting to `~/Pictures`.
fn watch_folder_from_env() -> PathBuf {
    match std::env::var_os("LUME_WATCH_FOLDER") {
        Some(path) => PathBuf::from(path),
        None => home_dir().join("Pictures"),
    }
}

fn spawn_sidecar() -> Option<Child> {
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
        SIDECAR_SOCKET_PATH.to_string(),
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

impl Drop for SidecarChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.lock().expect("sidecar child lock poisoned").take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod thumb_protocol_tests {
    use super::thumb_protocol_response;

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

        let resp = thumb_protocol_response(dir.path(), &request("lume://thumb/42"));

        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("Content-Type").unwrap(), "image/jpeg");
        assert_eq!(resp.body(), b"jpeg-bytes");
    }

    #[test]
    fn returns_404_for_a_missing_thumbnail() {
        let dir = tempfile::tempdir().unwrap();
        let resp = thumb_protocol_response(dir.path(), &request("lume://thumb/999"));
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
            let resp = thumb_protocol_response(dir.path(), &request(uri));
            assert_eq!(resp.status(), 400, "expected 400 for {uri}");
        }
    }
}
