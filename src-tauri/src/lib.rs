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

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use lume_core::Sidecar;
use lume_ipc::SocketSidecar;
use serde::{Deserialize, Serialize};
use tauri::Manager;

const M0_SOCKET_PATH: &str = "/tmp/lume-m0-sidecar.sock";

struct SidecarChild(Mutex<Option<Child>>);

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

/// M0 heartbeat: prove Rust can connect to the spawned Sidecar and get a vector.
#[tauri::command]
fn m0_sidecar_heartbeat() -> Result<usize, String> {
    SocketSidecar::new(M0_SOCKET_PATH)
        .embed_text("girl riding a bicycle")
        .map(|emb| emb.dim())
        .map_err(|err| err.to_string())
}

/// Build and run the Tauri application. Called by `main.rs`.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            app.manage(SidecarChild(Mutex::new(spawn_m0_sidecar())));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            search,
            start_index,
            index_status,
            m0_sidecar_heartbeat
        ])
        .run(tauri::generate_context!())
        .expect("error while running Lume");
}

fn spawn_m0_sidecar() -> Option<Child> {
    if std::env::var_os("LUME_DISABLE_M0_SIDECAR").is_some() {
        return None;
    }

    let sidecar_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri has a workspace parent")
        .join("sidecar");
    let child = Command::new("uv")
        .args([
            "run",
            "python",
            "-m",
            "lume_sidecar.server",
            "--socket",
            M0_SOCKET_PATH,
        ])
        .current_dir(sidecar_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    match child {
        Ok(child) => Some(child),
        Err(err) => {
            eprintln!("M0 Sidecar spawn skipped: {err}");
            None
        }
    }
}

impl Drop for SidecarChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.lock().expect("sidecar child lock poisoned").take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
