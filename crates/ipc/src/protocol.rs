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

/// Base64 ⇄ `Vec<u8>` serde adapter — the *one* place binary payload encoding
/// lives on the Rust side of the seam (issue #17, M2 transport depth).
///
/// DTO fields keep their `Vec<u8>` type so callers still see ordinary bytes,
/// but they cross the wire as compact base64 strings instead of JSON integer
/// arrays. Framing stays length-prefixed JSON, so frames remain inspectable.
mod b64 {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let encoded = String::deserialize(deserializer)?;
        STANDARD
            .decode(encoded.as_bytes())
            .map_err(serde::de::Error::custom)
    }
}

/// One request frame from Rust to Python.
///
/// The tag is the socket-level operation name. Keeping this envelope explicit
/// lets the Sidecar server multiplex bulk indexing, drag-in image queries, and
/// text queries on one length-prefixed transport without leaking model details.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum ClientMessage {
    Embed(EmbedRequest),
    EmbedOne(EmbedOneRequest),
    EmbedText(EmbedTextRequest),
}

/// One response frame from Python to Rust.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum ServerMessage {
    EmbedResponse(EmbedResponse),
    EmbedOneResponse(EmbedOneResponse),
    Error { message: String },
}

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
    /// Stored grid-thumbnail edge, px (`Config.thumbnails.grid_px`). A
    /// legitimate batch-level parameter, not a model/device/framework leak
    /// (DESIGN §19) — the sidecar still owns all decode/resize knowledge.
    pub thumb_px: u32,
    pub units: Vec<RequestUnit>,
}

/// One Unit's result. Per-unit failure is in-band, not fatal (DESIGN §17).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum UnitResult {
    Ok {
        /// 768 little-endian fp16 elements = 1536 bytes. Base64 on the wire.
        #[serde(with = "b64")]
        emb_fp16: Vec<u8>,
        /// Grid thumbnail, JPEG-encoded (DESIGN §8). Base64 on the wire.
        #[serde(with = "b64")]
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
    #[serde(with = "b64")]
    pub image_bytes: Vec<u8>,
}

/// Python → Rust: the drag-in query vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbedOneResponse {
    #[serde(with = "b64")]
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
