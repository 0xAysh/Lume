//! # lume-index (L3 ingest)
//!
//! [`Indexer`]: turns one watched folder of JPEG/PNG files into rows in the
//! store, streaming through the [`lume_core::Sidecar`] / [`lume_core::VectorStore`]
//! seams built in M1 Slices 1–2. M1 scope only (BUILD.md):
//!
//! - **Full re-index every run** — no per-file state machine, no FSEvents, no
//!   reconciliation, no basename-pairing (all M2, ADR-0002).
//! - EXIF/dimensions/GPS and eager BLAKE3 hashing stay unset — M4 and M2
//!   respectively.
//!
//! Metadata storage isn't one of the three named trait seams (`VectorStore`,
//! `Sidecar`, `Platform` — BUILD.md discipline checklist), so [`Indexer`] is
//! allowed to depend on the concrete [`lume_store::SqliteStore`] type for
//! metadata helpers while still going through `dyn Sidecar`/`VectorStore` for
//! the two seams that matter (workspace dependency direction: `app → index →
//! ipc → store → core`).

mod progress;
mod walk;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use lume_core::{
    EmbedOutcome, EmbedUnit, EmbeddedUnit, Embedding, FileId, IndexState, LumeError, MediaKind,
    Sidecar, VectorStore,
};
use lume_store::SqliteStore;

pub use progress::Progress;
pub use walk::walk_folder;

/// Walks one configured folder and embeds every JPEG/PNG into `store` via
/// `sidecar`, full-reset each run (M1 scope).
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

    /// Full re-index: reset the store, walk the folder, embed in
    /// `Config.batch_size`-sized batches. A single corrupt/unsupported file
    /// never aborts its batch (DESIGN §17) — it lands `IndexState::Failed`
    /// and indexing continues.
    pub fn run(&self) -> Result<(), LumeError> {
        self.store.reset_all()?;
        std::fs::create_dir_all(&self.thumbnails_dir)?;

        let paths = walk_folder(&self.root);
        self.progress.set_total(paths.len() as u64);

        for chunk in paths.chunks(self.batch_size) {
            self.run_batch(chunk)?;
            self.progress.add_done(chunk.len() as u64);
        }
        Ok(())
    }

    fn run_batch(&self, paths: &[PathBuf]) -> Result<(), LumeError> {
        let mut file_ids = Vec::with_capacity(paths.len());
        for path in paths {
            let metadata = std::fs::metadata(path)?;
            let mtime = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            file_ids.push(
                self.store
                    .upsert_file(path, MediaKind::Image, metadata.len(), mtime)?,
            );
        }

        let units: Vec<EmbedUnit> = paths
            .iter()
            .zip(&file_ids)
            .map(|(path, &file)| EmbedUnit {
                file,
                path: path.clone(),
                frame_ts: None,
            })
            .collect();

        let outcomes = self.sidecar.embed(&units)?;
        self.commit_outcomes(&file_ids, outcomes)
    }

    /// Writes thumbnails and marks per-file state; batches every successful
    /// embedding from this chunk into one [`VectorStore::insert_batch`] call
    /// so the chunk's Units become visible to search atomically.
    fn commit_outcomes(
        &self,
        file_ids: &[FileId],
        outcomes: Vec<EmbedOutcome>,
    ) -> Result<(), LumeError> {
        let mut embedded: Vec<(FileId, Embedding)> = Vec::new();

        for (&file_id, outcome) in file_ids.iter().zip(outcomes) {
            match outcome {
                EmbedOutcome::Ok {
                    emb,
                    thumbnail_jpeg,
                } => {
                    let thumb_path = self.thumbnails_dir.join(format!("{file_id}.jpg"));
                    std::fs::write(&thumb_path, &thumbnail_jpeg)?;
                    embedded.push((file_id, emb));
                }
                EmbedOutcome::Failed { reason } => {
                    tracing::warn!(file_id, %reason, "embedding failed, marking file Failed");
                    self.store
                        .set_file_state(file_id, IndexState::Failed, None)?;
                }
            }
        }

        if !embedded.is_empty() {
            let units: Vec<EmbeddedUnit<'_>> = embedded
                .iter()
                .map(|(file, emb)| EmbeddedUnit {
                    file: *file,
                    frame_ts: None,
                    emb,
                })
                .collect();
            self.store.insert_batch(&units)?;
            for (file_id, _) in &embedded {
                self.store
                    .set_file_state(*file_id, IndexState::Done, None)?;
            }
        }

        Ok(())
    }
}
