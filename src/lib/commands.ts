// The typed Tauri command seam (DESIGN §19: "UI holds ZERO business logic; only
// calls typed commands; never assumes backend internals"). Every backend call
// the UI makes goes through this file — the one place the L5↔L4 contract lives.
// These signatures mirror the `#[tauri::command]` handlers in
// `src-tauri/src/lib.rs`; keep the two in lockstep.

import { invoke } from "@tauri-apps/api/core";

/** One result Tile (DESIGN §12 / CONTEXT.md). Deduplicated from Units backend-side. */
export interface SearchHit {
  /** Item id (row in the `files` table). */
  fileId: number;
  /** `lume://` URL for the stored grid thumbnail / video poster (DESIGN §14). */
  thumbUrl: string;
  kind: "image" | "video";
  /** Tile score = max over the Item's matched Units (DESIGN §12). */
  score: number;
  /** Matched video-frame timestamps, for scrubber markers (DESIGN §7). Empty for images. */
  matchedTimestamps: number[];
}

/** Structured filters combined with the semantic query (DESIGN §12). */
export interface SearchFilters {
  kind?: "image" | "video";
  /** Unix seconds, inclusive. */
  capturedAfter?: number;
  capturedBefore?: number;
  folder?: string;
}

/** Coarse indexing lifecycle for the menu bar + onboarding (DESIGN §11, §18). */
export type IndexPhase = "idle" | "scanning" | "indexing" | "paused" | "error";

export interface IndexStatus {
  phase: IndexPhase;
  done: number;
  total: number;
}

/** Semantic search → ranked Tiles (DESIGN §12). */
export function search(query: string, filters?: SearchFilters): Promise<SearchHit[]> {
  return invoke<SearchHit[]>("search", { query, filters: filters ?? null });
}

/** Kick off (or resume) indexing of the watched folders. */
export function startIndex(): Promise<void> {
  return invoke<void>("start_index");
}

/** Poll current indexing progress. */
export function indexStatus(): Promise<IndexStatus> {
  return invoke<IndexStatus>("index_status");
}
