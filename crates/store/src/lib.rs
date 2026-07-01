//! # lume-store (L1 persistence)
//!
//! The sqlite + sqlite-vec adapter behind [`lume_core::VectorStore`], plus the
//! metadata store for [`lume_core::FileRecord`] rows. All the load-bearing
//! persistence discipline lives here, *behind* the trait:
//!
//! - **WAL mode + `busy_timeout`** at DB-open so search reads never block on the
//!   per-batch writer (DESIGN §10 "searchable as it progresses").
//! - **Single-writer funnel**: bulk index, FSEvents deltas, and search-history
//!   inserts all serialize through one writer path; searches use read-only
//!   connections (DESIGN §13 Tier 3, hardcoded).
//! - `vec_units` virtual table keyed `float16[768]` — *not* `float[768]` (§8).
//!
//! Schema sketch and the M0 storage proof: BUILD.md L1 and `docs/M0.md`.

// TODO(M0 decision): the Rust crate `sqlite-vec = 0.1.10-alpha.4` currently
//                    fails to build and its source has no float16 element type.
//                    Choose patch/vendor vs float32 vs int8 vs a different L1
//                    adapter before implementing production VectorStore.
// TODO(M1): implement VectorStore (insert_batch/filtered knn/delete_file) + MetadataStore
//           (upsert/list) with migrations from day one.
