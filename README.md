# Lume

Local-first semantic search over a personal photo/video library. Describe a
scene in plain language — *"a girl riding a bicycle"* — and Lume returns the
matching photos and video moments, ranked. Nothing leaves the machine.

macOS / Apple Silicon, v1. Tauri (Rust) shell + React/Vite UI + Python SigLIP 2
sidecar + sqlite-vec.

> Design rationale, build plan, glossary, and ADRs live in `docs/` (kept
> local-only). Start there: `docs/DESIGN.md` (what & why), `docs/BUILD.md`
> (how & in what order), `docs/CONTEXT.md` (the source-of-truth glossary).

## Layout

The architecture is layered; dependencies flow strictly downward and each layer
is its own crate, so the direction is enforced by the compiler (DESIGN §19).

```
crates/
  core/      L0  domain types · Config (single source) · LumeError · 3 trait seams
  store/     L1  sqlite + sqlite-vec  → VectorStore           (stub → M1)
  ipc/       L2  Unix-socket wire contract + Sidecar adapter  (contract now, I/O → M1)
  index/     L3  walk · pair · watch · reconcile · change-detect (stub → M2)
  media/     L3  ffmpeg frames · poster · lazy thumbnails      (stub → M3)
  platform/  ⊥   macOS power · thermal · paths · FSEvents       (stub → M5)
src-tauri/   L4  Tauri app: typed command surface (composition root)
src/         L5  React + Vite + TS frontend (zero business logic)
sidecar/         Python SigLIP black box (uv-managed) + pytest
```

The two load-bearing seams — the **socket contract** (`crates/ipc`) and
**single-source config** (`crates/core`) — must stay correct from commit one
(BUILD.md). Everything else is concrete until a second adapter actually exists.

## Prerequisites

- Rust (stable), Node 20+, [`uv`](https://docs.astral.sh/uv/) for the sidecar.
- `ffmpeg` (Homebrew) — needed from M3 (video) onward, not for the scaffold.

## Develop

```sh
make test      # cargo + pytest + tsc
make lint      # clippy -D warnings, fmt --check, ruff, tsc
make fmt       # auto-format all three languages
make dev       # run the app (Vite + Tauri)
```

First sidecar run: `cd sidecar && uv sync` (creates the pinned venv).

## Method

Test-driven, vertical slices (one test → one implementation). Tests verify
behavior through public interfaces, never internals. The deterministic cores
(result pipeline, basename-pairing, change detection, config, the wire codec)
are TDD'd directly; the GPU sidecar is faked behind its seam; signing/MPS/fp16
are de-risked in a spike (BUILD.md M0).

## Note: pinned `time` crate

`Cargo.lock` pins `time` to `0.3.51`. `time 0.3.52` changed `Parsable::parse`'s
signature, which breaks `cookie 0.18.1` (pulled transitively by Tauri/wry).
Remove the pin once `cookie` releases a compatible version.
