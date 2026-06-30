//! The Rust ↔ Python wire contract (DESIGN §9, §19 — BUILD.md L2).
//!
//! **Lock this early, never break it casually.** This is the load-bearing seam:
//! nothing about model / device / framework may ever appear in these messages.
//! If you find yourself adding a `device` or `model` field, the seam has leaked
//! — stop (DESIGN §19).
//!
//! Only *compressed/small* data crosses: a few KB of paths out, ~48 KB of fp16
//! vectors + ~1.3 MB of JPEG thumbnails back per 32-image batch. This is *why* a
//! plain Unix socket suffices and shared memory is rejected (§9). Raw pixels
//! never cross — the sidecar owns all decode (§6).
//!
//! These structs define the message *shape*. The concrete framing
//! (length-prefixed; JSON header + binary payload vs. msgpack) is finalized when
//! the transport lands — see `TODO(M1)` in `lib.rs`. Until then the round-trip
//! test pins the shape.

use serde::{Deserialize, Serialize};

/// One Unit to embed in a batch request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestUnit {
    /// Index within this batch — the response echoes it so results re-align even
    /// if the sidecar reorders internally.
    pub unit_idx: u32,
    pub path: String,
    /// `None` = whole image; `Some(ts)` = video frame at `ts` seconds.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub frame_ts: Option<f32>,
}

/// Rust → Python: embed a batch (DESIGN §9, bulk path).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub batch_id: u64,
    pub units: Vec<RequestUnit>,
}

/// One Unit's result. Per-unit failure is in-band, not fatal (DESIGN §17).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum UnitResult {
    Ok {
        /// 768 little-endian fp16 elements = 1536 bytes.
        emb_fp16: Vec<u8>,
        /// Grid thumbnail, JPEG-encoded (DESIGN §8).
        thumb_jpeg: Vec<u8>,
    },
    Failed {
        reason: String,
    },
}

/// Python → Rust: results for a batch, echoing `batch_id`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub batch_id: u64,
    pub items: Vec<BatchItem>,
}

/// A response item, carrying its `unit_idx` for re-alignment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchItem {
    pub unit_idx: u32,
    pub result: UnitResult,
}

/// Rust → Python: synchronous single-image embed for drag-in queries (§12, M4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbedOneRequest {
    pub image_bytes: Vec<u8>,
}

/// Python → Rust: the drag-in query vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbedOneResponse {
    pub emb_fp16: Vec<u8>,
}

/// Rust → Python: synchronous text-query embed for semantic search (§12).
///
/// The response reuses [`EmbedOneResponse`]: both interactive paths return one
/// bare query vector and no thumbnail.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbedTextRequest {
    pub text: String,
}
