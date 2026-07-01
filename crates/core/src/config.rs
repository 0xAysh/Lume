//! Single-source configuration (DESIGN §13).
//!
//! **The ONE place tunables live.** Every magic number in the system is a field
//! here; there are no scattered constants (BUILD.md discipline checklist). The
//! Preferences UI and Advanced section edit a *subset* of this same struct, then
//! Save commits it — which is what makes every tunable a one-line change.
//!
//! Persisted as `~/.lume/config.json`. [`Config::validate`] runs at Save so a
//! bad config never reaches the running system (§13 "Validation").

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;
use crate::types::{Dtype, OpenAction};

/// A watched root the walker scans recursively (DESIGN §5). Onboarding pre-fills
/// `~/Pictures` and `~/Movies`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FolderConfig {
    pub path: PathBuf,
    /// Per-folder opt-out without removing it (DESIGN §15 exclusions).
    #[serde(default = "default_true")]
    pub enabled: bool,
}

/// A walk-time ignore rule (glob or path prefix) — DESIGN §15.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExcludeRule {
    pub pattern: String,
}

/// Video frame-extraction bounds (DESIGN §7). All four are config-tunable; the
/// floor/ceiling/cap are what keep ~15k videos from exploding to millions of
/// embeddings — the load-bearing scale assumption.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VideoConfig {
    /// FFmpeg `select='gt(scene,THRESHOLD)'` trigger.
    pub scene_threshold: f32,
    /// Never sample two frames closer than this (kills rapid-cut explosions).
    pub floor_s: f32,
    /// Always sample at least one frame within this window (static videos stay
    /// visible). Must be strictly greater than `floor_s`.
    pub ceiling_s: f32,
    /// Hard cap per video — a 2-hour film maxes at this many frames.
    pub max_frames: u32,
    /// Bounded FFmpeg worker pool for lazy matched-frame extraction (§7).
    pub extract_pool_size: u32,
}

/// Thumbnail / preview sizing and the lazy cache cap (DESIGN §8).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThumbConfig {
    /// Stored grid thumbnail edge, px.
    pub grid_px: u32,
    /// On-demand full preview edge, px.
    pub preview_px: u32,
    /// Size-capped LRU for previews + matched video frames.
    pub cache_cap_bytes: u64,
}

/// Result-pipeline tuning (DESIGN §12). Cannot be reasoned to correct values a
/// priori — they depend on SigLIP 2 Base's real score distribution, so they
/// live here and are tuned empirically in M4 (tuning = settings change).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultConfig {
    /// Max Tiles returned — the *real* backstop against returning all 100k
    /// (the relevance floor is biased loose; the cap carries noise defense).
    pub tile_cap: usize,
    /// Relevance floor coefficient: keep Tiles scoring `>= alpha * top_score`.
    /// Relative, never an absolute cosine (§12). In `(0.0, 1.0]`.
    pub floor_alpha: f32,
    /// Minimum relative gap for the "strong / possibly related" cliff divider.
    /// No gap clears it → no divider (a designed outcome, not a failure).
    pub cliff_min_gap: f32,
    /// Over-fetch factor for KNN: `k = tile_cap * unit_oversample`, so dedup
    /// still yields a full grid when a few long videos dominate the top-k.
    pub unit_oversample: usize,
}

/// The whole configuration. `Default` is the shipped baseline; the UI mutates a
/// staged copy and commits on Save (DESIGN §13).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub watched_folders: Vec<FolderConfig>,

    /// SigLIP variant — swappable; changing it invalidates every embedding and
    /// forces a re-index (DESIGN §13 data-affecting settings).
    pub model_name: String,
    pub vector_dtype: Dtype,
    /// MPS batch size. Getting this wrong turns a ~1-hour index into days (§9).
    pub batch_size: usize,

    pub video: VideoConfig,
    pub thumbnails: ThumbConfig,
    pub results: ResultConfig,

    /// Safety-net reconciliation cadence (§10).
    pub reconcile_interval_s: u64,
    /// Live-search debounce (§12).
    pub debounce_ms: u64,
    /// Bulk indexing on battery — off by default (DESIGN §10 power policy).
    pub index_on_battery: bool,
    pub default_open_action: OpenAction,
    pub excludes: Vec<ExcludeRule>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            watched_folders: Vec::new(),
            model_name: "google/siglip2-base-patch16-224".to_string(),
            // float32 in the vec0 table (ADR-0003) — sqlite-vec has no float16
            // element type. The wire embedding stays fp16 (crates/ipc); only the
            // stored dtype flips.
            vector_dtype: Dtype::F32,
            batch_size: 32,
            video: VideoConfig {
                scene_threshold: 0.4,
                floor_s: 2.0,
                ceiling_s: 10.0,
                max_frames: 60,
                extract_pool_size: 4,
            },
            thumbnails: ThumbConfig {
                grid_px: 400,
                preview_px: 1200,
                cache_cap_bytes: 3 * 1024 * 1024 * 1024, // ~3 GB
            },
            results: ResultConfig {
                tile_cap: 500,
                floor_alpha: 0.65,
                cliff_min_gap: 0.05,
                unit_oversample: 6,
            },
            reconcile_interval_s: 6 * 60 * 60, // 6h
            debounce_ms: 275,
            index_on_battery: false,
            default_open_action: OpenAction::InApp,
            excludes: Vec::new(),
        }
    }
}

impl Config {
    /// Reject a config that would break the running system. Run at Save (§13),
    /// so the in-memory config is always valid.
    ///
    /// Checks the invariants a UI slider can violate; it does not verify that
    /// watched folders *exist* (an external drive may be unplugged — that's a
    /// `Stale` runtime condition, §17, not a config error).
    pub fn validate(&self) -> Result<(), ConfigError> {
        let v = &self.video;
        if v.floor_s >= v.ceiling_s {
            return Err(ConfigError::Invalid {
                field: "video.floor_s",
                reason: format!(
                    "floor ({}) must be strictly less than ceiling ({})",
                    v.floor_s, v.ceiling_s
                ),
            });
        }
        if v.max_frames == 0 {
            return Err(invalid("video.max_frames", "must be at least 1"));
        }
        if v.extract_pool_size == 0 {
            return Err(invalid("video.extract_pool_size", "must be at least 1"));
        }
        if v.scene_threshold <= 0.0 {
            return Err(invalid("video.scene_threshold", "must be > 0"));
        }
        if self.batch_size == 0 {
            return Err(invalid("batch_size", "must be at least 1"));
        }
        if self.model_name.trim().is_empty() {
            return Err(invalid("model_name", "must not be empty"));
        }

        let r = &self.results;
        if r.floor_alpha <= 0.0 || r.floor_alpha > 1.0 {
            return Err(invalid(
                "results.floor_alpha",
                "must be in (0.0, 1.0] — it is a fraction of the top score",
            ));
        }
        if r.tile_cap == 0 {
            return Err(invalid("results.tile_cap", "must be at least 1"));
        }
        if r.unit_oversample == 0 {
            return Err(invalid(
                "results.unit_oversample",
                "must be at least 1 (k = tile_cap * unit_oversample)",
            ));
        }
        if r.cliff_min_gap < 0.0 {
            return Err(invalid("results.cliff_min_gap", "must be >= 0"));
        }

        if self.thumbnails.grid_px == 0 || self.thumbnails.preview_px == 0 {
            return Err(invalid("thumbnails", "grid_px and preview_px must be > 0"));
        }

        Ok(())
    }
}

fn invalid(field: &'static str, reason: &str) -> ConfigError {
    ConfigError::Invalid {
        field,
        reason: reason.to_string(),
    }
}

fn default_true() -> bool {
    true
}
