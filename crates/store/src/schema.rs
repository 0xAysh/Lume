//! Schema migrations, applied via `PRAGMA user_version` (BUILD.md discipline
//! checklist: "migrations from day one," not ad-hoc `CREATE TABLE IF NOT
//! EXISTS` calls scattered through the code).
//!
//! `files` holds one row per **Item** (ADR-0002 — never one row per on-disk
//! File); `units` holds one row per embedded **Unit**; `vec_units` is the
//! sqlite-vec virtual table keyed by `units.id`, storing `float[768]`
//! (float32, ADR-0003 — sqlite-vec has no float16 element type).

use lume_core::LumeError;
use rusqlite::Connection;

/// Ordered SQL steps. Step `i` (0-indexed) takes the DB from
/// `user_version = i` to `user_version = i + 1`. Append, never edit, once a
/// step has shipped and could exist in a real user's `~/.lume`.
const MIGRATIONS: &[&str] = &[r#"
    CREATE TABLE files (
        id          INTEGER PRIMARY KEY,
        path        TEXT NOT NULL UNIQUE,
        kind        INTEGER NOT NULL,
        size        INTEGER NOT NULL,
        mtime       INTEGER NOT NULL,
        hash        BLOB,
        state       INTEGER NOT NULL,
        captured_at INTEGER,
        width       INTEGER NOT NULL,
        height      INTEGER NOT NULL,
        duration_s  REAL,
        folder      TEXT NOT NULL,
        gps_lat     REAL,
        gps_lng     REAL
    );

    CREATE TABLE units (
        id       INTEGER PRIMARY KEY,
        file_id  INTEGER NOT NULL REFERENCES files(id),
        frame_ts REAL
    );
    CREATE INDEX units_file_id_idx ON units(file_id);

    -- Exact brute-force KNN (DESIGN §8). float32 per ADR-0003; cosine is
    -- recovered from L2 distance over the sidecar's L2-normalized vectors
    -- (score = 1 - distance^2 / 2), so no distance_metric=cosine column.
    CREATE VIRTUAL TABLE vec_units USING vec0(embedding float[768]);
"#];

/// Bring `conn`'s schema up to the latest version. Idempotent — safe to call
/// on every `SqliteStore::open`.
pub fn apply(conn: &Connection) -> Result<(), LumeError> {
    let current_version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(store_err)?;
    let current_version = current_version.max(0) as usize;

    for (i, step) in MIGRATIONS.iter().enumerate().skip(current_version) {
        conn.execute_batch(step).map_err(store_err)?;
        conn.pragma_update(None, "user_version", (i + 1) as i64)
            .map_err(store_err)?;
    }
    Ok(())
}

fn store_err(e: rusqlite::Error) -> LumeError {
    LumeError::Store(e.to_string())
}
