# Lume

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-macOS%20%28Apple%20Silicon%29-lightgrey)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)

**Local-first semantic search over your photo and video library.** Describe a
scene in plain language — *"a girl riding a bicycle"* — and Lume returns the
matching photos and video moments, ranked. No cloud, no accounts, no
telemetry: nothing ever leaves the machine.

Lume is a Tauri (Rust) desktop app for macOS on Apple Silicon, with a
React/Vite UI and a Python sidecar running SigLIP 2 for embeddings. Vectors
are stored and searched with `sqlite-vec` using exact (brute-force) k-NN —
no approximate-search accuracy loss at this scale.

> **Status:** pre-alpha. The workspace is scaffolded and the core seams
> (config, error types, the sidecar wire protocol) are under active
> development. There is no working build yet — see [Roadmap](#roadmap).

---

## Why

Most photo search is either cloud-based (your photos leave your machine) or
keyword/EXIF-based (it can't find "a girl riding a bicycle" unless someone
tagged it that way). Lume runs a vision-language model locally so you can
search by *meaning*, entirely offline, over your own files.

The product stance is **recall over precision**: a few near-miss results cost
you a glance; a missing result is total feature failure. Every ranking and
cutoff decision in the design follows from that.

## Features (target, v1)

- **Natural-language search** over photos and videos via SigLIP 2 embeddings.
- **Exact k-NN**, not approximate — perfect recall at this library scale.
- **Video moment search**: matches land on the specific frame/timestamp, not
  just "this video somewhere."
- **Identity-aware indexing**: Live Photos and RAW+JPEG pairs are indexed as
  one item, not duplicated.
- **Fully local**: no network calls, no accounts, no telemetry.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| App shell | Tauri (Rust) | Native macOS window via WKWebView — no bundled Chromium. |
| UI | React + Vite + TypeScript | Lightweight SPA; the UI holds zero business logic. |
| Inference | Python sidecar, SigLIP 2 on PyTorch/MPS | No mature Rust binding for PyTorch-on-MPS exists today. |
| Vector store | `sqlite-vec` | Exact brute-force k-NN, single file, no daemon. |
| Metadata | SQLite | File state, index progress, search history. |
| Video | FFmpeg | Scene-change detection and bounded frame extraction. |
| File watching | `notify` (FSEvents) | Live updates to watched folders. |

## Architecture

The codebase is layered; dependencies flow strictly downward, and each layer
is its own crate so the direction is enforced by the compiler.

```
crates/
  core/      L0  domain types · config (single source) · error types · trait seams
  store/     L1  sqlite + sqlite-vec  → VectorStore
  ipc/       L2  Unix-socket wire contract + sidecar adapter
  index/     L3  walk · pair · watch · reconcile · change detection
  media/     L3  ffmpeg frames · poster frames · lazy thumbnails
  platform/  ⊥   macOS power · thermal · paths · FSEvents
src-tauri/   L4  Tauri app — typed command surface (composition root)
src/         L5  React + Vite + TS frontend (zero business logic)
sidecar/         Python SigLIP process (uv-managed) + pytest
```

The two seams that have to be right from day one are the **socket contract**
(`crates/ipc`) and the **single-source config** (`crates/core`); everything
above them stays free to change until a second concrete implementation
actually exists.

## Getting started

### Prerequisites

- macOS on Apple Silicon
- Rust (stable)
- Node 20+
- [`uv`](https://docs.astral.sh/uv/) for the Python sidecar
- `ffmpeg` (via Homebrew) — required from the video milestone onward

### Setup

```sh
git clone https://github.com/0xAysh/Lume.git
cd Lume
npm install
cd sidecar && uv sync && cd ..
```

### Develop

```sh
make dev      # run the app (Vite + Tauri)
make test     # run every test suite (cargo + pytest + tsc)
make lint     # clippy -D warnings, fmt --check, ruff, tsc
make fmt      # auto-format all three languages
```

Run `make help` for the full list of targets.

## Roadmap

| Milestone | Outcome |
|---|---|
| M0 — Spine spike | Tauri ↔ Python sidecar over a socket, real SigLIP embedding on MPS, signed `.app` launches. |
| M1 — Walking skeleton | One folder, JPEG only, full index → search → results grid, end to end. |
| M2 — Ingest depth | HEIC/RAW, batched pipeline, resumable state, FSEvents + reconciliation. |
| M3 — Video | Scene detection, frame embeddings, one tile per video, lazy frame extraction. |
| M4 — Query/UX | Adaptive relevance cutoff, virtualized grid, filters, find-similar. **← MVP** |
| M5 — Lifecycle | Menu bar, close-to-tray, idle unload, power/thermal policy. |
| M6 — Settings | Staged settings, hot reload, consolidated re-index prompts. |
| M7 — Robustness | Failure surfacing, sidecar respawn, onboarding, logging. |
| M8 — Packaging | Signed/notarized bundle, model weight download + checksum. |

Currently in **M0**. Full rationale and detailed task breakdowns are tracked
internally and as [issues](https://github.com/0xAysh/Lume/issues) on this
repo.

## Method

Test-driven, vertical slices — one test, then its implementation, never
all-tests-then-all-code. Tests exercise public interfaces, never internals,
so they survive refactors. The deterministic cores (search ranking,
basename-pairing, change detection, config, the wire codec) are TDD'd
directly; the GPU sidecar is faked behind its seam in tests that don't need
a real model; signing and MPS/fp16 behavior are de-risked with a spike.

## Known issues

`Cargo.lock` pins the `time` crate to `0.3.51`. `time 0.3.52` changed
`Parsable::parse`'s signature, which breaks `cookie 0.18.1` (pulled
transitively by Tauri/wry). Remove the pin once `cookie` ships a compatible
release.

## Contributing

Issues are tracked on GitHub (`0xAysh/Lume`). This project isn't yet open
for external contributions while the core architecture is being established
— feel free to open an issue for bugs or ideas in the meantime.

## License

[MIT](LICENSE) © Ayush Rangrej
