//! # lume-index (L3 ingest)
//!
//! [`Indexer`]: turns one watched folder of JPEG/PNG files into rows in the
//! store, streaming through the [`lume_core::Sidecar`] / [`lume_core::VectorStore`]
//! seams built in M1 Slices 1–2. M1 scope only (BUILD.md):
//!
//! - **Per-Item resume** — stable `Done`/`Failed` Items are skipped, while
//!   `Pending` Items resume after a restart. FSEvents, reconciliation, and
//!   basename-pairing remain later M2 slices (ADR-0002).
//! - Eager BLAKE3 hashes are persisted for `Done` Items and used as the
//!   move/rename tiebreaker; EXIF/dimensions/GPS stay unset until M4.
//!
//! Metadata storage isn't one of the three named trait seams (`VectorStore`,
//! `Sidecar`, `Platform` — BUILD.md discipline checklist), so [`Indexer`] is
//! allowed to depend on the concrete [`lume_store::SqliteStore`] type for
//! metadata helpers while still going through `dyn Sidecar`/`VectorStore` for
//! the two seams that matter (workspace dependency direction: `app → index →
//! ipc → store → core`).

mod commit;
mod progress;
mod walk;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use lume_core::{Blake3Hash, EmbedUnit, LumeError, MediaKind, Sidecar};
use lume_store::SqliteStore;

pub use commit::BatchCommitter;
pub use progress::Progress;
pub use walk::walk_folder;

/// Walks one configured folder and embeds every pending supported still-image
/// Item into `store` via `sidecar`.
pub struct Indexer {
    root: PathBuf,
    batch_size: usize,
    thumbnails_dir: PathBuf,
    store: Arc<SqliteStore>,
    sidecar: Arc<dyn Sidecar + Send + Sync>,
    progress: Arc<Progress>,
}

impl Indexer {
    pub fn new(
        root: PathBuf,
        batch_size: usize,
        thumbnails_dir: PathBuf,
        store: Arc<SqliteStore>,
        sidecar: Arc<dyn Sidecar + Send + Sync>,
    ) -> Self {
        Self {
            root,
            batch_size: batch_size.max(1),
            thumbnails_dir,
            store,
            sidecar,
            progress: Arc::new(Progress::default()),
        }
    }

    /// A handle callers can poll (e.g. Tauri's `index_status` command,
    /// Slice 4) independently of whatever thread [`Self::run`] executes on.
    pub fn progress(&self) -> Arc<Progress> {
        Arc::clone(&self.progress)
    }

    /// Resumable indexing: walk the folder, skip stable `Done`/`Failed` Items,
    /// and embed remaining `Pending` work in `Config.batch_size`-sized batches.
    /// A single corrupt/unsupported file never aborts its batch (DESIGN §17) —
    /// it lands `IndexState::Failed` and indexing continues.
    pub fn run(&self) -> Result<(), LumeError> {
        std::fs::create_dir_all(&self.thumbnails_dir)?;

        let paths = walk_folder(&self.root);
        let seen_paths: BTreeSet<PathBuf> = paths.iter().cloned().collect();
        self.progress.set_total(paths.len() as u64);

        for chunk in paths.chunks(self.batch_size) {
            self.run_batch(chunk, &seen_paths)?;
            self.progress.add_done(chunk.len() as u64);
        }
        self.store
            .delete_missing_files_under_root(&self.root, &seen_paths)?;
        Ok(())
    }

    fn run_batch(
        &self,
        paths: &[PathBuf],
        seen_paths: &BTreeSet<PathBuf>,
    ) -> Result<(), LumeError> {
        let mut file_ids = Vec::with_capacity(paths.len());
        let mut hashes = Vec::with_capacity(paths.len());
        let mut work_paths = Vec::with_capacity(paths.len());
        for path in paths {
            let metadata = std::fs::metadata(path)?;
            let mtime = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let hash = hash_item(path, metadata.len())?;
            if let Some(file_id) = self.store.prepare_file_for_index(
                path,
                MediaKind::Image,
                metadata.len(),
                mtime,
                hash,
                seen_paths,
            )? {
                file_ids.push(file_id);
                hashes.push(hash);
                work_paths.push(path.clone());
            }
        }

        if file_ids.is_empty() {
            return Ok(());
        }

        let units: Vec<EmbedUnit> = work_paths
            .iter()
            .zip(&file_ids)
            .map(|(path, &file)| EmbedUnit {
                file,
                path: path.clone(),
                frame_ts: None,
            })
            .collect();

        let outcomes = self.sidecar.embed(&units)?;

        // Delegate the whole "embedded Item becomes visible" transaction —
        // thumbnails, the atomic Unit insert, and per-Item state — to the one
        // module that owns those commit semantics (see [`commit`]). The store
        // plays both roles: the `VectorStore` seam for the atomic insert and
        // the concrete metadata helper for state.
        BatchCommitter::new(
            &self.thumbnails_dir,
            self.store.as_ref(),
            self.store.as_ref(),
        )
        .commit(&file_ids, &hashes, outcomes)
    }
}

fn hash_item(path: &Path, size: u64) -> Result<Blake3Hash, LumeError> {
    // Still images are small enough to hash fully. The chunked branch documents
    // the large-media policy M2 needs before M3 video decoding exists: use a
    // bounded first+last sample plus size, not a multi-hour full-file read.
    const LARGE_MEDIA_THRESHOLD: u64 = 512 * 1024 * 1024;
    const EDGE_CHUNK: u64 = 1024 * 1024;

    let mut hasher = blake3::Hasher::new();
    if size <= LARGE_MEDIA_THRESHOLD {
        hasher.update(&std::fs::read(path)?);
    } else {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = std::fs::File::open(path)?;
        let mut first = vec![0_u8; EDGE_CHUNK as usize];
        let first_len = file.read(&mut first)?;
        hasher.update(&first[..first_len]);

        let tail_start = size.saturating_sub(EDGE_CHUNK);
        file.seek(SeekFrom::Start(tail_start))?;
        let mut last = vec![0_u8; EDGE_CHUNK as usize];
        let last_len = file.read(&mut last)?;
        hasher.update(&last[..last_len]);
        hasher.update(&size.to_le_bytes());
    }

    Ok(Blake3Hash(*hasher.finalize().as_bytes()))
}
