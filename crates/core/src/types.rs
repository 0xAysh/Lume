//! Domain identity types.
//!
//! Vocabulary is fixed by `docs/CONTEXT.md` — keep these names exact:
//! **File** (on disk) → **Item** (one tile, what Lume indexes) → **Unit** (one
//! embeddable input → one vector). A [`FileRecord`] is one row in the `files`
//! table, i.e. one **Item** (its primary File); Companions are skipped at walk
//! time and never get a row (ADR-0002: "index items, not files").

use std::fmt;
use std::path::PathBuf;

use half::f16;
use serde::{Deserialize, Serialize};

/// Row id of an **Item** in the metadata store.
pub type FileId = i64;

/// A BLAKE3 digest (32 bytes), the move/rename tiebreaker in change detection
/// (DESIGN §10). Fixed 32 bytes regardless of file size, so 100k items ≈ 3.2 MB.
/// Computed *eagerly* at index time and persisted before any move can occur.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Blake3Hash(pub [u8; 32]);

impl fmt::Debug for Blake3Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Whether an Item is a still image or a video. Drives frame extraction (§7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaKind {
    Image,
    Video,
}

/// Per-Item index lifecycle (DESIGN §10 crash-safety, §17 stale folders).
///
/// `Done` carries a [`Blake3Hash`]; `Pending`/`Failed` may not yet (the read
/// that computes the hash hasn't happened). `Stale` = the file's folder is
/// currently unreachable (external drive) — *not* deleted (DESIGN §17).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexState {
    Pending,
    Done,
    Failed,
    Stale,
}

/// The 768-dim **Embedding** (CONTEXT.md) SigLIP maps an image/frame or a query
/// text into. Stored as fp16 (DESIGN §8). Image and text land in the same space,
/// which is what makes search work.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding(pub Vec<f16>);

impl Embedding {
    /// SigLIP 2 Base dimensionality. The vector dtype is config-swappable
    /// (DESIGN §13) but the dimension is fixed by the model.
    pub const DIM: usize = 768;

    /// Length of the underlying vector. May differ from [`Self::DIM`] only if a
    /// future model variant changes dimensionality (config swap + re-index).
    pub fn dim(&self) -> usize {
        self.0.len()
    }
}

/// One row in the `files` table: one **Item**, keyed by its primary File.
///
/// Companions (the `.MOV` of a Live Photo, the RAW beside a JPEG) do not appear
/// here — the walker's basename-pairing pass collapses them (DESIGN §5).
#[derive(Clone, Debug, PartialEq)]
pub struct FileRecord {
    pub id: FileId,
    pub path: PathBuf,
    pub kind: MediaKind,
    pub size: u64,
    pub mtime: i64,
    /// Eager move-detection hash. `None` only while `Pending`/`Failed` (§10).
    pub hash: Option<Blake3Hash>,
    pub state: IndexState,

    // Metadata extracted at index time, nearly free (DESIGN §12 filters).
    pub captured_at: Option<i64>,
    pub width: u32,
    pub height: u32,
    /// Present for videos only.
    pub duration_s: Option<f32>,
    pub folder: PathBuf,
    pub gps: Option<(f64, f64)>,
}

/// A single embeddable input → one vector (CONTEXT.md: **Unit**).
///
/// Media-agnostic so image and video frames share one embedding queue (§9):
/// `frame_ts == None` is a whole image; `Some(ts)` is a video frame at `ts`
/// seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbedUnit {
    pub file: FileId,
    pub path: PathBuf,
    pub frame_ts: Option<f32>,
}

/// One embedded **Unit** ready to be committed to the vector store.
///
/// A batch of these is the transaction boundary for indexing: either every Unit
/// in the batch is visible to search, or none is. That invariant belongs behind
/// [`crate::VectorStore`], not in every caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbeddedUnit<'a> {
    pub file: FileId,
    pub frame_ts: Option<f32>,
    pub emb: &'a Embedding,
}

/// Structured filters combined with semantic KNN (DESIGN §12).
///
/// These are pushed below the vector-store seam so the sqlite-vec adapter can
/// join against metadata while searching. Filtering only after top-k would make
/// date/type/folder filters amputate otherwise-good grids.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchFilters {
    pub kind: Option<MediaKind>,
    pub captured_after: Option<i64>,
    pub captured_before: Option<i64>,
    pub folder: Option<PathBuf>,
}

/// A scored **Unit** from a KNN query — one image Unit or one video frame.
///
/// The result pipeline (DESIGN §12) collapses these by `file` into **Tiles**
/// *before* the floor/cap/cliff run; never sort/cut on raw `ScoredHit`s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredHit {
    pub file: FileId,
    pub frame_ts: Option<f32>,
    /// Cosine similarity; higher is closer. Poorly calibrated across queries
    /// (DESIGN §12) — only ever compared *within* one query's result set.
    pub score: f32,
}

/// Default click action for a result tile (DESIGN §12, configurable).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAction {
    /// In-app preview/player.
    InApp,
    /// Hand off to Preview.app / QuickTime.
    DefaultApp,
}

/// Stored vector element type (DESIGN §8, §13 — config-swappable; changing it
/// invalidates every embedding and forces a re-index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dtype {
    F16,
    F32,
}
