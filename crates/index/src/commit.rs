//! The one local surface that owns "an embedded Item becomes visible."
//!
//! M1's `Indexer` wrote Tile thumbnails, called [`VectorStore::insert_batch`],
//! and flipped `files.state` to `Done` as three inline steps. M2 adds resume,
//! per-Item state, and crash safety (DESIGN §10), so those steps need to live
//! behind one interface an agent can test in isolation: given a batch of
//! embedding outcomes, exactly what ends up on disk, in the vector store, and
//! in `files.state` when a thumbnail write fails, a vector insert fails, or an
//! embedding failed in-band.
//!
//! The commit order is the load-bearing invariant: an Item's thumbnail and its
//! Unit vector rows are both durable *before* its state flips to `Done`. So a
//! `Done` Item is never missing its Tile and never missing a searchable Unit —
//! the two states M2 resume must be able to trust.
//!
//! Seam discipline (BUILD.md): the atomic Unit insert goes through the
//! `VectorStore` trait (so a fake seam can be injected under test), while the
//! `files.state` transitions stay on the concrete [`SqliteStore`] metadata
//! helpers — metadata storage is deliberately *not* a fourth trait seam.

use std::path::Path;

use lume_core::{
    Blake3Hash, EmbedOutcome, EmbeddedUnit, Embedding, FileId, IndexState, LumeError, VectorStore,
};
use lume_store::SqliteStore;

/// Owns the commit transaction for one batch of embedding outcomes: Tile
/// thumbnails on disk, Unit vector rows through [`VectorStore::insert_batch`],
/// and the per-Item `files.state` flip to `Done`.
///
/// Constructed per batch from borrowed handles. In production `vectors` and
/// `state` are the same [`SqliteStore`] viewed through its two roles; tests can
/// inject a failing `VectorStore` while keeping a real store for state.
pub struct BatchCommitter<'a> {
    thumbnails_dir: &'a Path,
    vectors: &'a dyn VectorStore,
    state: &'a SqliteStore,
}

impl<'a> BatchCommitter<'a> {
    pub fn new(
        thumbnails_dir: &'a Path,
        vectors: &'a dyn VectorStore,
        state: &'a SqliteStore,
    ) -> Self {
        Self {
            thumbnails_dir,
            vectors,
            state,
        }
    }

    /// Commit one batch of `outcomes`, positionally aligned with `file_ids`.
    ///
    /// Per-Item problems never abort the batch (DESIGN §17): an in-band
    /// embedding failure, or a thumbnail write error, marks *only* that Item
    /// [`IndexState::Failed`] and moves on. Every Item whose thumbnail landed
    /// is inserted through the single [`VectorStore::insert_batch`] transaction
    /// — the batch's crash-resume commit boundary — and only after that atomic
    /// insert succeeds do those Items flip to [`IndexState::Done`].
    ///
    /// A vector-store error is therefore surfaced (not swallowed) with every
    /// Item in the batch left un-`Done` and resumable, rather than `Done` yet
    /// unsearchable. Any thumbnails already written are harmless orphans that a
    /// later re-embed overwrites.
    pub fn commit(
        &self,
        file_ids: &[FileId],
        hashes: &[Blake3Hash],
        outcomes: Vec<EmbedOutcome>,
    ) -> Result<(), LumeError> {
        let mut embedded: Vec<(FileId, Blake3Hash, Embedding)> = Vec::new();

        for ((&file_id, &hash), outcome) in file_ids.iter().zip(hashes).zip(outcomes) {
            match outcome {
                EmbedOutcome::Ok {
                    emb,
                    thumbnail_jpeg,
                } => {
                    let thumb_path = self.thumbnails_dir.join(format!("{file_id}.jpg"));
                    match std::fs::write(&thumb_path, &thumbnail_jpeg) {
                        Ok(()) => embedded.push((file_id, hash, emb)),
                        Err(err) => {
                            tracing::warn!(
                                file_id,
                                %err,
                                "thumbnail write failed, marking Item Failed"
                            );
                            self.state
                                .set_file_state(file_id, IndexState::Failed, None)?;
                        }
                    }
                }
                EmbedOutcome::Failed { reason } => {
                    tracing::warn!(file_id, %reason, "embedding failed, marking Item Failed");
                    self.state
                        .set_file_state(file_id, IndexState::Failed, None)?;
                }
            }
        }

        if embedded.is_empty() {
            return Ok(());
        }

        let units: Vec<EmbeddedUnit<'_>> = embedded
            .iter()
            .map(|(file, _, emb)| EmbeddedUnit {
                file: *file,
                frame_ts: None,
                emb,
            })
            .collect();

        // Batch visibility is owned by this single transaction. Only once it
        // commits do the Items flip to Done, so a failure here can never leave
        // a Done Item without a searchable Unit.
        self.vectors.insert_batch(&units)?;
        for (file_id, hash, _) in &embedded {
            self.state
                .set_file_state(*file_id, IndexState::Done, Some(*hash))?;
        }
        Ok(())
    }
}
