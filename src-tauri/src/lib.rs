//! # lume-app (L4 application)
//!
//! The Tauri shell and the **typed command surface** the UI calls. All adapter
//! construction and live runtime state live in [`runtime::AppRuntime`]; this file
//! is the *thin* command seam DESIGN §19 asks for ("app/ = Tauri commands, wiring
//! (thin)"). Each command translates DTOs and delegates to the runtime — it holds
//! no lifecycle, indexing, or path-resolution logic.
//!
//! The DTOs are webview-facing (camelCase JSON) and mirror `src/lib/commands.ts`.
//! Keep the two in lockstep. The UI never sees the runtime internals (DESIGN §19:
//! "UI holds zero business logic").

mod runtime;

use std::path::PathBuf;

use lume_core::rank::Tile;
use lume_core::MediaKind;
use runtime::{AppRuntime, RunPhase};
use serde::{Deserialize, Serialize};
use tauri::Manager;

/// One result Tile (DESIGN §12). Deduplicated from Units before it crosses here.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub file_id: i64,
    /// `lume://` URL for the stored grid thumbnail / video poster (DESIGN §14).
    pub thumb_url: String,
    pub kind: HitKind,
    pub score: f32,
    /// Matched video-frame timestamps for scrubber markers (DESIGN §7).
    pub matched_timestamps: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HitKind {
    Image,
    Video,
}

/// Structured filters combined with the semantic query (DESIGN §12).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchFilters {
    pub kind: Option<HitKindFilter>,
    pub captured_after: Option<i64>,
    pub captured_before: Option<i64>,
    pub folder: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HitKindFilter {
    Image,
    Video,
}

/// Coarse indexing lifecycle for the menu bar + onboarding (DESIGN §11, §18).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexStatus {
    pub phase: IndexPhase,
    pub done: u64,
    pub total: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexPhase {
    Idle,
    Scanning,
    Indexing,
    Paused,
    Error,
}

/// Semantic search → ranked Tiles (DESIGN §12). Translates the filter DTO in and
/// the [`Tile`]s out; the over-fetch → collapse → floor → cap → cliff pipeline is
/// owned by [`AppRuntime::search`], not this seam.
#[tauri::command]
fn search(
    runtime: tauri::State<AppRuntime>,
    query: String,
    filters: Option<SearchFilters>,
) -> Result<Vec<SearchHit>, String> {
    let tiles = runtime.search(&query, to_core_filters(filters))?;
    Ok(tiles.iter().map(to_search_hit).collect())
}

/// Explicit reconciliation hook for startup/manual safety-net scans. The
/// concurrency guard and run bookkeeping live in [`AppRuntime::start_reconcile`].
#[tauri::command]
fn reconcile_now(runtime: tauri::State<AppRuntime>) -> Result<(), String> {
    runtime.start_reconcile()
}

/// Kick off (or resume) indexing of the watched folder. Kept as the M1 UI
/// command alias; it now delegates to the same reconcile path.
#[tauri::command]
fn start_index(runtime: tauri::State<AppRuntime>) -> Result<(), String> {
    runtime.start_index()
}

/// Poll current indexing progress. [`AppRuntime::index_status`] owns the phase
/// state machine; this seam maps it onto the webview `IndexPhase` DTO.
#[tauri::command]
fn index_status(runtime: tauri::State<AppRuntime>) -> IndexStatus {
    let status = runtime.index_status();
    IndexStatus {
        phase: match status.phase {
            RunPhase::Idle => IndexPhase::Idle,
            RunPhase::Scanning => IndexPhase::Scanning,
            RunPhase::Indexing => IndexPhase::Indexing,
        },
        done: status.done,
        total: status.total,
    }
}

fn to_core_filters(filters: Option<SearchFilters>) -> lume_core::SearchFilters {
    let filters = filters.unwrap_or_default();
    lume_core::SearchFilters {
        kind: filters.kind.map(|k| match k {
            HitKindFilter::Image => MediaKind::Image,
            HitKindFilter::Video => MediaKind::Video,
        }),
        captured_after: filters.captured_after,
        captured_before: filters.captured_before,
        folder: filters.folder.map(PathBuf::from),
    }
}

fn to_search_hit(tile: &Tile) -> SearchHit {
    SearchHit {
        file_id: tile.file,
        thumb_url: format!("lume://thumb/{}", tile.thumb_id),
        kind: match tile.kind {
            MediaKind::Image => HitKind::Image,
            MediaKind::Video => HitKind::Video,
        },
        score: tile.score,
        matched_timestamps: tile.matched_timestamps.clone(),
    }
}

/// Build and run the Tauri application. Called by `main.rs`. All adapter wiring
/// and live state is owned by [`AppRuntime`]; the protocol handler and command
/// handlers are thin delegations into it.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .register_uri_scheme_protocol("lume", |ctx, request| {
            ctx.app_handle()
                .state::<AppRuntime>()
                .thumbnail_response(&request)
        })
        .setup(|app| {
            app.manage(AppRuntime::bootstrap());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            search,
            reconcile_now,
            start_index,
            index_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running Lume");
}
