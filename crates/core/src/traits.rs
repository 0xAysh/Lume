//! The three trait seams (DESIGN §19).
//!
//! Each has **exactly one adapter today** and a **named** second adapter on the
//! v2 roadmap. That is what makes them *earned* seams rather than speculation
//! (BUILD.md: "One adapter = hypothetical seam. Two adapters = real seam" — these
//! qualify because the second adapter is concretely on the roadmap):
//!
//! | Trait         | Today          | Named v2 second adapter        |
//! |---------------|----------------|--------------------------------|
//! | [`VectorStore`] | sqlite-vec   | usearch/HNSW past ~1M vectors  |
//! | [`Sidecar`]     | PyTorch/MPS  | ONNX / Core ML (Python-only)   |
//! | [`Platform`]    | macOS        | Windows, then Linux            |

use std::path::PathBuf;

use crate::error::LumeError;
use crate::types::{EmbedUnit, EmbeddedUnit, Embedding, FileId, ScoredHit, SearchFilters};

/// L1 seam: exact brute-force KNN over fp16 vectors.
///
/// The interface is deliberately small but carries the load-bearing store
/// obligations: batch commits are atomic, filtering is pushed into the metadata
/// join, and all sqlite-vec/WAL/single-writer machinery (DESIGN §10) lives
/// *behind* it. Callers never see SQL or transaction handles.
pub trait VectorStore {
    /// Atomically insert one committed indexing batch.
    ///
    /// This is the transaction boundary from DESIGN §10. The Indexer hands over
    /// a batch; the adapter owns WAL, busy timeouts, single-writer funneling,
    /// and the exact commit point used for crash resume.
    fn insert_batch(&self, units: &[EmbeddedUnit<'_>]) -> Result<(), LumeError>;

    /// Exact k-nearest **Units**, with structured filters applied in-store.
    ///
    /// Returns up to `k` [`ScoredHit`]s, best first. Collapsing to Tiles happens
    /// above this seam (DESIGN §12), but date/type/folder filters belong here so
    /// the adapter can apply them before/with the KNN scan instead of trimming
    /// an already-small top-k after the fact.
    fn knn(
        &self,
        query: &Embedding,
        k: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<ScoredHit>, LumeError>;

    /// Remove every Unit belonging to an Item (e.g. file deleted).
    fn delete_file(&self, file: FileId) -> Result<(), LumeError>;
}

/// L2 seam — **the load-bearing one**. The sidecar is a BLACK BOX:
/// `path → (embedding, thumbnail)`. NOTHING about model / device / framework
/// crosses this interface, which is what keeps the entire ONNX / Core ML /
/// CUDA future a Python-only change (DESIGN §19, §21).
pub trait Sidecar {
    /// Batch-embed Units. Returns one [`EmbedOutcome`] per input, in order.
    /// Per-unit failures are reported in-band (not as `Err`) — a corrupt photo
    /// must not abort the batch (DESIGN §17).
    fn embed(&self, units: &[EmbedUnit]) -> Result<Vec<EmbedOutcome>, LumeError>;

    /// Synchronous single-image embed for drag-in queries (DESIGN §12, M4).
    /// Distinct from the bulk [`Self::embed`] path; pays the vision-tower reload
    /// if it was unloaded on idle (§11).
    fn embed_one(&self, image: &[u8]) -> Result<Embedding, LumeError>;

    /// Synchronous text-query embed for semantic search (DESIGN §12).
    ///
    /// This is the hottest interactive path and is served by the resident text
    /// tower (§11). Like [`Self::embed_one`], it bypasses bulk indexing and
    /// should be drained between image batches by the sidecar adapter.
    fn embed_text(&self, query: &str) -> Result<Embedding, LumeError>;
}

/// Result of embedding one Unit. Success carries the vector *and* the grid
/// thumbnail (generated for free from the same decode, DESIGN §6, §8).
#[derive(Clone, Debug)]
pub enum EmbedOutcome {
    Ok {
        emb: Embedding,
        thumbnail_jpeg: Vec<u8>,
    },
    /// → [`crate::IndexState::Failed`]; surfaced in the "couldn't index" UI (§17).
    Failed { reason: String },
}

/// Thermal pressure level reported by the OS (DESIGN §10 power/thermal policy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalLevel {
    Nominal,
    Fair,
    Serious,
    Critical,
}

/// A filesystem change observed by the platform watcher (DESIGN §10 FSEvents).
#[derive(Clone, Debug, PartialEq)]
pub enum FsEvent {
    Created(PathBuf),
    Modified(PathBuf),
    Removed(PathBuf),
}

/// Callback the [`Platform`] watcher pushes [`FsEvent`]s into.
pub type EventSink = Box<dyn Fn(FsEvent) + Send + Sync>;

/// Cross-cutting seam: everything OS-specific (power, thermal, paths, watching)
/// lives behind this one trait so porting is "add one adapter," never a
/// sprinkle of `#[cfg(target_os)]` (DESIGN §19, §21).
pub trait Platform {
    /// True when on AC power — gates bulk indexing (DESIGN §10).
    fn on_ac_power(&self) -> bool;

    /// Current thermal pressure; throttle batch rate when high.
    fn thermal_pressure(&self) -> ThermalLevel;

    /// Lume's data directory (`~/.lume` on macOS) — DESIGN §8.
    fn data_dir(&self) -> PathBuf;

    /// Begin watching `roots` recursively, pushing changes to `sink`
    /// (FSEvents-backed on macOS).
    fn watch(&self, roots: &[PathBuf], sink: EventSink) -> Result<(), LumeError>;
}
