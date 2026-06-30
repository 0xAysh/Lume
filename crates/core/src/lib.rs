//! # lume-core (L0 foundation)
//!
//! Domain types, the single-source [`Config`], typed [`LumeError`]s, and the
//! three trait seams ([`VectorStore`], [`Sidecar`], [`Platform`]) that every
//! other layer depends on. This crate has no knowledge of SQLite, sockets,
//! FFmpeg, or Tauri — dependencies flow strictly *toward* it (BUILD.md layer
//! map). See `docs/DESIGN.md` §19 for the seam rationale and `docs/CONTEXT.md`
//! for the File → Item → Unit → Tile vocabulary these types encode.

pub mod config;
pub mod error;
pub mod traits;
pub mod types;

pub use config::{Config, ExcludeRule, FolderConfig, ResultConfig, ThumbConfig, VideoConfig};
pub use error::{ConfigError, LumeError};
pub use traits::{EmbedOutcome, EventSink, FsEvent, Platform, Sidecar, ThermalLevel, VectorStore};
pub use types::{
    Blake3Hash, Dtype, EmbedUnit, EmbeddedUnit, Embedding, FileId, FileRecord, IndexState,
    MediaKind, OpenAction, ScoredHit, SearchFilters,
};
