//! Behavioural tests for the single-source [`Config`] (DESIGN §13 "Validation").
//!
//! These exercise the public interface only (`Config::default`, `validate`,
//! serde round-trip) — no internals — so they survive any refactor of how
//! validation is implemented.

// These tests deliberately start from the valid baseline and corrupt one field
// to assert that exactly that invariant is enforced.
#![allow(clippy::field_reassign_with_default)]

use lume_core::{Config, ConfigError};

#[test]
fn default_config_is_valid() {
    // The shipped baseline must always pass its own validation.
    assert!(Config::default().validate().is_ok());
}

#[test]
fn video_floor_must_be_below_ceiling() {
    let mut cfg = Config::default();
    cfg.video.floor_s = 12.0;
    cfg.video.ceiling_s = 10.0;

    let err = cfg.validate().unwrap_err();
    assert!(
        matches!(err, ConfigError::Invalid { field, .. } if field == "video.floor_s"),
        "expected a video.floor_s invariant error, got {err:?}",
    );
}

#[test]
fn relevance_floor_alpha_must_be_a_fraction() {
    // alpha is `>= alpha * top_score`, so values outside (0, 1] are nonsense.
    for bad in [0.0_f32, -0.1, 1.5] {
        let mut cfg = Config::default();
        cfg.results.floor_alpha = bad;
        assert!(
            cfg.validate().is_err(),
            "floor_alpha = {bad} should be rejected",
        );
    }
}

#[test]
fn batch_size_zero_is_rejected() {
    let mut cfg = Config::default();
    cfg.batch_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_round_trips_through_json() {
    // The Preferences UI persists this exact struct to ~/.lume/config.json.
    let cfg = Config::default();
    let json = serde_json::to_string_pretty(&cfg).expect("serialize");
    let back: Config = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg, back);
}
