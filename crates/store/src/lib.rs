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
//! Schema sketch and the M0 fp16-under-concurrent-append proof: BUILD.md L1.

// TODO(M0): pin the sqlite-vec version; prove float16[768] KNN reads 2-byte
//           elements and stays correct under WAL while a batch is appended.
// TODO(M1): implement VectorStore (insert_batch/filtered knn/delete_file) + MetadataStore
//           (upsert/list) with migrations from day one.
