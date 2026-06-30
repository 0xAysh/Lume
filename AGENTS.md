# Lume

Local-first semantic search over a personal photo/video library. macOS / Apple Silicon, v1.
Tauri (Rust) shell + React/Vite UI + Python SigLIP 2 sidecar + sqlite-vec.

Start from `docs/DESIGN.md` (what & why) and `docs/BUILD.md` (how & in what order).
Glossary in `docs/CONTEXT.md` is the source of truth for terminology.

## Agent skills

### Issue tracker

GitHub Issues on `0xAysh/Lume`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical five roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`),
plus `prd` and `milestone:M0…M8`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `docs/CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.
