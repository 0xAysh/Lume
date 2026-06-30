//! # lume-app (L4 application)
//!
//! The Tauri shell and the **typed command surface** the UI calls. This is the
//! composition root: as the engine fills in, it constructs the [`VectorStore`],
//! [`Sidecar`], and [`Platform`] adapters and wires them into the commands here.
//! The UI never sees any of that — it only invokes the commands below
//! (DESIGN §19: "Tauri commands = typed API; UI holds zero business logic").
//!
//! The DTOs are webview-facing (camelCase JSON) and mirror `src/lib/commands.ts`.
//! Keep the two in lockstep.

use serde::{Deserialize, Serialize};

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

/// Semantic search → ranked Tiles (DESIGN §12).
#[tauri::command]
fn search(query: String, filters: Option<SearchFilters>) -> Vec<SearchHit> {
    // TODO(M1): embed the query via the Sidecar → knn over the VectorStore →
    //           collapse Units to Tiles. TODO(M4): adaptive cutoff + cliff.
    let _ = (query, filters);
    Vec::new()
}

/// Kick off (or resume) indexing of the watched folders.
#[tauri::command]
fn start_index() {
    // TODO(M1): hand off to the Indexer (walk → Sidecar → store).
}

/// Poll current indexing progress.
#[tauri::command]
fn index_status() -> IndexStatus {
    // TODO(M1): report real progress from the Indexer.
    IndexStatus {
        phase: IndexPhase::Idle,
        done: 0,
        total: 0,
    }
}

/// Build and run the Tauri application. Called by `main.rs`.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![search, start_index, index_status])
        .run(tauri::generate_context!())
        .expect("error while running Lume");
}
