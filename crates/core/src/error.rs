//! Typed errors that cross seams.
//!
//! BUILD.md discipline checklist: "no stringly errors across seams." Every
//! fallible trait method in [`crate::traits`] returns [`LumeError`], so callers
//! match on a variant instead of parsing a message string.

use std::path::PathBuf;

/// The one error type that crosses the L0 trait seams.
///
/// Per-*unit* embedding failures are **not** modelled here — a corrupt photo is
/// expected, not exceptional, so it is reported in-band as
/// [`crate::traits::EmbedOutcome::Failed`] and surfaced in the "couldn't index"
/// UI (DESIGN §17). `LumeError` is for failures that abort the *operation*.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LumeError {
    /// Underlying I/O failure (socket, filesystem, DB file).
    #[error("io error: {0}")]
    Io(String),

    /// The vector store (sqlite-vec) rejected an operation.
    #[error("vector store error: {0}")]
    Store(String),

    /// The sidecar transport failed — socket dead, framing corrupt, or the
    /// process is gone. Distinct from a per-unit embed failure (DESIGN §17:
    /// Rust auto-respawns the sidecar on this class of error).
    #[error("sidecar transport error: {0}")]
    Sidecar(String),

    /// A [`crate::Config`] failed validation at load or Save (DESIGN §13).
    #[error("invalid config: {0}")]
    Config(#[from] ConfigError),

    /// A requested item/path is not present in the store.
    #[error("not found: {0}")]
    NotFound(String),
}

impl From<std::io::Error> for LumeError {
    fn from(e: std::io::Error) -> Self {
        LumeError::Io(e.to_string())
    }
}

/// Why a [`crate::Config`] is invalid. Validation runs at Save so a bad config
/// never reaches the running system (DESIGN §13 "Validation").
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// e.g. the video frame floor is not strictly below the ceiling (§7).
    #[error("{field}: {reason}")]
    Invalid { field: &'static str, reason: String },

    /// A watched/excluded path is empty or malformed.
    #[error("bad path: {0}")]
    BadPath(PathBuf),
}
