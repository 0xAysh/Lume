//! # lume-index (L3 ingest)
//!
//! Turns a watched-folder tree into a stream of **Items** to embed, and keeps
//! the store in sync as the library changes. Thickens *behind* the stable
//! [`lume_core::Sidecar`] / [`lume_core::VectorStore`] seams — no interface
//! above L3 should change as this fills in (BUILD.md M2).
//!
//! Responsibilities (DESIGN §5, §10):
//! - **Walk** the tree, applying the **basename-pairing pass** so a Live Photo
//!   and a RAW+JPEG pair each collapse to *one* Item (ADR-0002), Companions
//!   skipped.
//! - **Change detection**: `(path, size, mtime, hash, embedding_id, state)`;
//!   mtime fast-path + eager BLAKE3 hash as the move/rename tiebreaker.
//! - **FSEvents** live updates + periodic **reconciliation** safety net.
//! - **Per-file state machine** `Pending → Done | Failed`, resume-safe.
//! - **Newest-first** ordering so recent photos are searchable within minutes.

// TODO(M2): basename-pairing pass (pure filename logic — prime TDD target).
// TODO(M2): change-detection state machine (pure transitions — prime TDD target).
// TODO(M2): wire walk → Sidecar → VectorStore/MetadataStore; FSEvents + reconcile.
