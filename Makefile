# Lume developer tasks across the three languages (Rust engine, Python sidecar,
# TS/React UI). CI and humans run the same targets.

.PHONY: help test test-rust test-py test-ui lint fmt build dev clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

test: test-rust test-py test-ui ## Run every test suite

test-rust: ## Rust workspace tests
	cargo test --workspace

test-py: ## Python sidecar tests
	cd sidecar && uv run pytest

test-ui: ## Frontend typecheck (the UI's "test" until component tests land)
	npm run typecheck

lint: ## Lint everything (no writes)
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --all -- --check
	cd sidecar && uv run ruff check .
	npm run typecheck

fmt: ## Auto-format everything
	cargo fmt --all
	cd sidecar && uv run ruff format . && uv run ruff check --fix .

build: ## Release build of engine + frontend
	cargo build --workspace --release
	npm run build

dev: ## Run the app in dev mode (Vite + Tauri)
	npm run tauri dev

clean: ## Remove build artifacts
	cargo clean
	rm -rf dist node_modules sidecar/.venv
