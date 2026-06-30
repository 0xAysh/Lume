//! # lume-ipc (L2 inference seam)
//!
//! The Rust side of the [`lume_core::Sidecar`] black box: the wire [`protocol`]
//! and (soon) the socket adapter that streams paths to Python and consumes
//! vectors + thumbnails. The Python process owns all decode, preprocess, and
//! embedding; this crate only frames bytes (DESIGN §6, §9).
//!
//! The wire contract is scaffolded now because it is one of the two seams that
//! "must be correct from commit one" (BUILD.md). The transport is not.

pub mod protocol;

// TODO(M1): finalize length-prefixed framing over a 0600 Unix socket; implement
//           the `Sidecar` adapter (`embed`, `embed_one`); auto-respawn on a dead
//           socket and resume from the last committed batch (DESIGN §17).
