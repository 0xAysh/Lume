//! # lume-media (L3 media)
//!
//! Wraps FFmpeg so version/flag changes touch one file (DESIGN §19). Note the
//! sidecar decodes *pixels for embedding* (§6); this crate handles *video frame
//! geometry and thumbnail/preview production* around that.
//!
//! Responsibilities (DESIGN §7, §8 — ADR-0001):
//! - **Scene detection** `select='gt(scene,THRESHOLD)'` bounded by 2s floor /
//!   10s ceiling / 60-frame cap (all config-tunable).
//! - **Poster harvest**: one ~400px poster per video, free from the scene-detect
//!   decode (highest-scene-score frame) — every tile paints instantly.
//! - **Lazy matched-frame extraction** via a bounded worker pool (≈3–4) — never
//!   one FFmpeg process per visible tile; LRU-cached.
//! - **Lazy 1200px previews** generated on click, LRU-cached.

// TODO(M3): scene detection + bounding + poster harvest.
// TODO(M3): bounded matched-frame extraction worker pool + LRU cache.
// TODO(M4): lazy preview generation.
