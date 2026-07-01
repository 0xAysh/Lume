//! Result pipeline: scored **Units** → ranked **Tiles** (DESIGN §12).
//!
//! This is the one place the Lume result contract lives, testable without
//! booting Tauri, starting the Sidecar, or touching sqlite-vec. The L4 `search`
//! command over-fetches Units from [`crate::VectorStore::knn`] with
//! `k = tile_cap * unit_oversample` and hands the raw [`ScoredHit`]s here; this
//! module owns collapse, floor, cap, and cliff so ranking never leaks across the
//! command surface.
//!
//! The order is load-bearing (DESIGN §12, "units → tiles → cutoff"):
//!
//! 1. Over-fetch Units (the *caller's* job — the KNN `k`).
//! 2. Collapse Units to Tiles by Item/File id.
//! 3. Tile rank score = `max(score)` across its matched Units (best-evidence;
//!    breadth-bonus re-rank is deferred to v2, never smuggled in here).
//! 4. Retain **all** matched timestamps for the Item, deterministically ordered
//!    (score desc, then ts asc). M1 images have none; videos need scrubber marks.
//! 5. Relative relevance floor on **Tile** scores: keep `>= floor_alpha * top`.
//! 6. Tile cap, applied **after** the floor.
//! 7. Optional cliff divider, detected **only within** the already-returned
//!    Tiles. If no gap clears `cliff_min_gap`, there is no divider. The cliff is
//!    never a cutoff.

use std::collections::HashMap;

use crate::config::ResultConfig;
use crate::types::{FileId, MediaKind, ScoredHit};

/// Item-level metadata the ranker needs to finish a [`Tile`], resolved per
/// `FileId`. Decouples ranking from storage: M1 supplies `Image` + the file id
/// as its own thumbnail identity; M3 supplies `Video` + a poster-frame identity
/// (ADR-0001) without any change to this module or the UI contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileMeta {
    pub kind: MediaKind,
    /// Identity of the stored grid thumbnail / video poster the L4 command turns
    /// into a `lume://thumb/<id>` URL (DESIGN §14). Usually the `FileId` itself.
    pub thumb_id: FileId,
}

/// One ranked, user-facing result Tile (DESIGN §12 / CONTEXT.md). Deduplicated
/// from Units; carries every matched timestamp for the Item, not just the best.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    pub file: FileId,
    pub kind: MediaKind,
    pub thumb_id: FileId,
    /// `max(score)` across the Item's matched Units.
    pub score: f32,
    /// Every matched Unit timestamp for the Item, score-desc then ts-asc.
    /// Empty for images (one whole-image Unit, `frame_ts == None`).
    pub matched_timestamps: Vec<f32>,
}

/// The optional "strong / possibly related" divider (DESIGN §12). Never a
/// cutoff: it only marks *where within the returned Tiles* the largest
/// significant score gap falls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cliff {
    /// Number of Tiles above the divider — the divider is drawn *after*
    /// `tiles[above - 1]`, so `tiles[..above]` are "strong".
    pub above: usize,
    /// The relative gap `score[above-1] - score[above]` that triggered it.
    pub gap: f32,
}

/// Output of the result pipeline: the ranked Tile grid plus an optional cliff.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedTiles {
    pub tiles: Vec<Tile>,
    pub cliff: Option<Cliff>,
}

/// Collapse raw KNN Unit hits into the ranked Tile grid (DESIGN §12).
///
/// `meta` resolves each `FileId` to its [`MediaKind`] and thumbnail identity.
/// The caller is responsible only for the over-fetch `k`; every collapse/cutoff
/// rule lives here. See the module docs for the load-bearing step order.
pub fn rank_tiles<F>(hits: Vec<ScoredHit>, config: &ResultConfig, meta: F) -> RankedTiles
where
    F: Fn(FileId) -> TileMeta,
{
    // 2. Collapse Units to Tiles by Item/File id. Accumulate the max score and
    //    every matched timestamp per file. HashMap order is non-deterministic;
    //    the sort below re-imposes a total order.
    struct Acc {
        score: f32,
        stamps: Vec<(f32, f32)>, // (ts, unit score)
    }
    let mut by_file: HashMap<FileId, Acc> = HashMap::new();
    for hit in hits {
        let acc = by_file.entry(hit.file).or_insert(Acc {
            score: hit.score,
            stamps: Vec::new(),
        });
        // 3. Tile rank score = max(score) across its Units.
        if hit.score > acc.score {
            acc.score = hit.score;
        }
        // 4. Retain all matched timestamps (images have `None` and contribute
        //    nothing here, so image Tiles keep an empty list).
        if let Some(ts) = hit.frame_ts {
            acc.stamps.push((ts, hit.score));
        }
    }

    let mut tiles: Vec<Tile> = by_file
        .into_iter()
        .map(|(file, mut acc)| {
            // Deterministic timestamp order: score desc, then ts asc.
            acc.stamps
                .sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));
            let m = meta(file);
            Tile {
                file,
                kind: m.kind,
                thumb_id: m.thumb_id,
                score: acc.score,
                matched_timestamps: acc.stamps.into_iter().map(|(ts, _)| ts).collect(),
            }
        })
        .collect();

    // Best-first, deterministic on ties by ascending file id.
    tiles.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.file.cmp(&b.file))
    });

    // 5. Relative relevance floor on TILE scores: keep `>= floor_alpha * top`.
    if let Some(top) = tiles.first().map(|t| t.score) {
        let floor = config.floor_alpha * top;
        tiles.retain(|t| t.score >= floor);
    }

    // 6. Tile cap, AFTER the floor.
    tiles.truncate(config.tile_cap);

    // 7. Cliff: the largest gap between adjacent returned Tiles, if it clears
    //    `cliff_min_gap`. Never a cutoff — the tile list is already final.
    let cliff = detect_cliff(&tiles, config.cliff_min_gap);

    RankedTiles { tiles, cliff }
}

/// Largest adjacent score gap within the returned Tiles, if any clears
/// `min_gap`. Ties on gap size resolve to the earliest (highest) divider.
fn detect_cliff(tiles: &[Tile], min_gap: f32) -> Option<Cliff> {
    let mut best: Option<Cliff> = None;
    for i in 0..tiles.len().saturating_sub(1) {
        let gap = tiles[i].score - tiles[i + 1].score;
        if gap >= min_gap && best.is_none_or(|b| gap > b.gap) {
            best = Some(Cliff { above: i + 1, gap });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ResultConfig {
        ResultConfig {
            tile_cap: 500,
            floor_alpha: 0.65,
            cliff_min_gap: 0.05,
            unit_oversample: 6,
        }
    }

    fn image_meta(id: FileId) -> TileMeta {
        TileMeta {
            kind: MediaKind::Image,
            thumb_id: id,
        }
    }

    fn image_hit(file: FileId, score: f32) -> ScoredHit {
        ScoredHit {
            file,
            frame_ts: None,
            score,
        }
    }

    fn frame_hit(file: FileId, ts: f32, score: f32) -> ScoredHit {
        ScoredHit {
            file,
            frame_ts: Some(ts),
            score,
        }
    }

    #[test]
    fn empty_input_yields_empty_tiles_and_no_cliff() {
        let out = rank_tiles(Vec::new(), &config(), image_meta);
        assert!(out.tiles.is_empty());
        assert!(out.cliff.is_none());
    }

    #[test]
    fn single_image_unit_becomes_one_tile_with_no_timestamps() {
        let out = rank_tiles(vec![image_hit(7, 0.9)], &config(), image_meta);
        assert_eq!(out.tiles.len(), 1);
        let tile = &out.tiles[0];
        assert_eq!(tile.file, 7);
        assert_eq!(tile.kind, MediaKind::Image);
        assert_eq!(tile.thumb_id, 7);
        assert_eq!(tile.score, 0.9);
        assert!(tile.matched_timestamps.is_empty());
        assert!(out.cliff.is_none());
    }

    #[test]
    fn duplicate_units_from_one_item_collapse_via_max_score() {
        // Same file id across three Units: one Tile, score = max.
        let hits = vec![
            frame_hit(1, 0.0, 0.4),
            frame_hit(1, 2.0, 0.8),
            frame_hit(1, 4.0, 0.6),
        ];
        let out = rank_tiles(hits, &config(), |id| TileMeta {
            kind: MediaKind::Video,
            thumb_id: id,
        });
        assert_eq!(
            out.tiles.len(),
            1,
            "three Units of one Item collapse to one Tile"
        );
        assert_eq!(out.tiles[0].score, 0.8, "Tile score is max over its Units");
        // Retained timestamps, ordered score-desc then ts-asc: 2.0(0.8),4.0(0.6),0.0(0.4)
        assert_eq!(out.tiles[0].matched_timestamps, vec![2.0, 4.0, 0.0]);
    }

    #[test]
    fn floor_is_applied_after_collapse_to_tile_scores() {
        // Item 1 has a weak Unit (0.3) and a strong Unit (1.0) -> Tile score 1.0,
        // survives the floor. A raw-Unit floor would have cut the 0.3 Unit and is
        // the bug this guards against. Item 2 is a genuinely weak Tile, cut.
        let cfg = ResultConfig {
            floor_alpha: 0.65,
            ..config()
        };
        let hits = vec![
            frame_hit(1, 0.0, 0.3),
            frame_hit(1, 2.0, 1.0),
            image_hit(2, 0.5), // 0.5 < 0.65 * 1.0 -> dropped
        ];
        let out = rank_tiles(hits, &cfg, image_meta);
        assert_eq!(out.tiles.len(), 1);
        assert_eq!(out.tiles[0].file, 1);
        assert_eq!(out.tiles[0].score, 1.0);
    }

    #[test]
    fn cap_is_applied_after_the_floor() {
        // Five Tiles all clear the floor (loose alpha), but cap = 2 trims to 2.
        let cfg = ResultConfig {
            tile_cap: 2,
            floor_alpha: 0.1,
            ..config()
        };
        let hits = vec![
            image_hit(1, 0.99),
            image_hit(2, 0.98),
            image_hit(3, 0.97),
            image_hit(4, 0.96),
            image_hit(5, 0.95),
        ];
        let out = rank_tiles(hits, &cfg, image_meta);
        assert_eq!(
            out.tiles.len(),
            2,
            "cap trims after the floor keeps them all"
        );
        assert_eq!(out.tiles[0].file, 1);
        assert_eq!(out.tiles[1].file, 2);
    }

    #[test]
    fn smooth_ramp_returns_no_cliff() {
        // Even gaps of 0.02, none clears cliff_min_gap = 0.05.
        let cfg = ResultConfig {
            floor_alpha: 0.1,
            cliff_min_gap: 0.05,
            ..config()
        };
        let hits = vec![
            image_hit(1, 0.30),
            image_hit(2, 0.28),
            image_hit(3, 0.26),
            image_hit(4, 0.24),
        ];
        let out = rank_tiles(hits, &cfg, image_meta);
        assert_eq!(out.tiles.len(), 4);
        assert!(
            out.cliff.is_none(),
            "no gap clears the threshold -> no divider"
        );
    }

    #[test]
    fn a_detected_gap_returns_cliff_divider_metadata() {
        // Two strong, a 0.30 gap, then two weak. Cliff divides after tile 2.
        let cfg = ResultConfig {
            floor_alpha: 0.1,
            cliff_min_gap: 0.05,
            ..config()
        };
        let hits = vec![
            image_hit(1, 0.90),
            image_hit(2, 0.88),
            image_hit(3, 0.58),
            image_hit(4, 0.56),
        ];
        let out = rank_tiles(hits, &cfg, image_meta);
        assert_eq!(out.tiles.len(), 4, "cliff never removes results");
        let cliff = out.cliff.expect("a 0.30 gap must produce a divider");
        assert_eq!(cliff.above, 2);
        assert!((cliff.gap - 0.30).abs() < 1e-6);
    }

    #[test]
    fn cliff_is_detected_only_within_returned_tiles() {
        // A big gap exists in the tail, but the cap cuts before it, so no cliff.
        let cfg = ResultConfig {
            tile_cap: 2,
            floor_alpha: 0.1,
            cliff_min_gap: 0.05,
            ..config()
        };
        let hits = vec![
            image_hit(1, 0.90),
            image_hit(2, 0.89),
            image_hit(3, 0.10), // big gap, but trimmed by cap before cliff runs
        ];
        let out = rank_tiles(hits, &cfg, image_meta);
        assert_eq!(out.tiles.len(), 2);
        assert!(out.cliff.is_none());
    }
}
