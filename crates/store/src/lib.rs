//! # lume-store (L1 persistence)
//!
//! [`SqliteStore`]: the sqlite + sqlite-vec adapter behind
//! [`lume_core::VectorStore`], plus the metadata helpers the Indexer needs for
//! [`lume_core::FileRecord`] rows. All the load-bearing persistence discipline
//! lives here, *behind* the trait:
//!
//! - **WAL mode + `busy_timeout`** at DB-open so search reads never block on
//!   the per-batch writer (DESIGN §10 "searchable as it progresses").
//! - **Single-writer funnel**: one long-lived writer connection behind a
//!   [`Mutex`]; [`VectorStore::knn`] opens a fresh read-only connection per
//!   call so search never blocks on the writer and vice versa.
//! - `vec_units` keyed `float[768]` (float32, ADR-0003 — sqlite-vec has no
//!   float16 element type; the wire embedding stays fp16, see `lume-ipc`).
//! - **Migrations from day one** via `PRAGMA user_version` ([`schema`]).
//!
//! Schema sketch and the storage decision record: BUILD.md L1 and
//! `docs/adr/0003-vector-storage-float32-sqlite-vec.md`.

mod schema;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};
use std::time::Duration;

use lume_core::{
    Blake3Hash, EmbeddedUnit, Embedding, FileId, FileRecord, IndexState, LumeError, MediaKind,
    ScoredHit, SearchFilters, VectorStore,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension};

static REGISTER_SQLITE_VEC: Once = Once::new();

/// Register the sqlite-vec extension with SQLite's `sqlite3_auto_extension`
/// hook. Process-global and must happen before *any* connection is opened
/// (writer or reader) — `Once` makes repeated calls (one per `SqliteStore`,
/// one per reader) safe.
///
/// The `bundled` `rusqlite` feature is required: without it,
/// `sqlite3_auto_extension` registers against a different SQLite than the one
/// `rusqlite` opens connections against, and `vec_version()`/`vec0` fail at
/// query time with no build error (ADR-0003).
fn register_sqlite_vec() {
    REGISTER_SQLITE_VEC.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut std::ffi::c_char,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> std::ffi::c_int,
        >(
            sqlite_vec::sqlite3_vec_init as *const ()
        )));
    });
}

/// The concrete L1 [`VectorStore`] adapter, plus the file-metadata helpers
/// the Indexer needs. Metadata storage is deliberately *not* a fourth named
/// trait seam (BUILD.md discipline checklist) — it's concrete here.
pub struct SqliteStore {
    db_path: PathBuf,
    writer: Mutex<Connection>,
}

impl SqliteStore {
    /// Open (creating if absent) the database at `db_path`, applying schema
    /// migrations and connection pragmas.
    pub fn open(db_path: impl Into<PathBuf>) -> Result<Self, LumeError> {
        let db_path = db_path.into();
        let writer = open_writer(&db_path)?;
        Ok(Self {
            db_path,
            writer: Mutex::new(writer),
        })
    }

    fn reader(&self) -> Result<Connection, LumeError> {
        open_reader(&self.db_path)
    }

    /// Insert or update the `files` row for `path` (unique key), returning
    /// its [`FileId`]. Always (re)sets `state` to [`IndexState::Pending`] —
    /// M1 has no incremental re-index, so an upsert means "about to embed
    /// this again." EXIF/dimensions/hash are M2/M4 concerns and stay
    /// unset (BUILD.md M1 scope).
    pub fn upsert_file(
        &self,
        path: &Path,
        kind: MediaKind,
        size: u64,
        mtime: i64,
    ) -> Result<FileId, LumeError> {
        let folder = path.parent().unwrap_or_else(|| Path::new(""));
        let conn = self.writer.lock().expect("writer lock poisoned");
        conn.query_row(
            "INSERT INTO files (path, kind, size, mtime, state, width, height, folder)
             VALUES (?1, ?2, ?3, ?4, ?5, 0, 0, ?6)
             ON CONFLICT(path) DO UPDATE SET
                kind = excluded.kind,
                size = excluded.size,
                mtime = excluded.mtime,
                state = excluded.state,
                folder = excluded.folder
             RETURNING id",
            rusqlite::params![
                path_to_text(path),
                kind_to_i64(kind),
                size as i64,
                mtime,
                state_to_i64(IndexState::Pending),
                path_to_text(folder),
            ],
            |row| row.get(0),
        )
        .map_err(store_err)
    }

    /// Ensure a walked Item has a metadata row and decide whether it still
    /// needs embedding for this run.
    ///
    /// M2 resume trusts stable `Done`/`Failed` Items: if path, size, and mtime
    /// are unchanged, the Indexer skips them instead of embedding again. A
    /// `Pending` Item is resumable work. If the file changed, old Units are
    /// discarded and the Item goes back to `Pending`; deeper rename/delete
    /// reconciliation lands in later M2 slices.
    pub fn prepare_file_for_index(
        &self,
        path: &Path,
        kind: MediaKind,
        size: u64,
        mtime: i64,
        hash: Blake3Hash,
        seen_paths: &BTreeSet<PathBuf>,
    ) -> Result<Option<FileId>, LumeError> {
        let folder = path.parent().unwrap_or_else(|| Path::new(""));
        let mut conn = self.writer.lock().expect("writer lock poisoned");
        let existing = conn
            .query_row(
                "SELECT id, size, mtime, hash, state FROM files WHERE path = ?1",
                [path_to_text(path)],
                |row| {
                    let hash: Option<Vec<u8>> = row.get(3)?;
                    Ok((
                        row.get::<_, FileId>(0)?,
                        row.get::<_, i64>(1)? as u64,
                        row.get::<_, i64>(2)?,
                        hash.map(|bytes| Blake3Hash(bytes_to_hash_array(&bytes))),
                        i64_to_state(row.get(4)?),
                    ))
                },
            )
            .optional()
            .map_err(store_err)?;

        match existing {
            Some((id, existing_size, existing_mtime, existing_hash, state))
                if existing_size == size
                    && existing_mtime == mtime
                    && existing_hash == Some(hash) =>
            {
                match state {
                    IndexState::Pending => Ok(Some(id)),
                    IndexState::Done | IndexState::Failed | IndexState::Stale => Ok(None),
                }
            }
            Some((id, _, _, _, _)) => {
                let tx = conn.transaction().map_err(store_err)?;
                delete_units_for_file(&tx, id)?;
                tx.execute(
                    "UPDATE files
                     SET kind = ?1, size = ?2, mtime = ?3, hash = ?4, state = ?5, folder = ?6
                     WHERE id = ?7",
                    rusqlite::params![
                        kind_to_i64(kind),
                        size as i64,
                        mtime,
                        hash.0.to_vec(),
                        state_to_i64(IndexState::Pending),
                        path_to_text(folder),
                        id,
                    ],
                )
                .map_err(store_err)?;
                tx.commit().map_err(store_err)?;
                Ok(Some(id))
            }
            None => {
                if move_done_file_by_hash(&conn, path, kind, size, mtime, hash, seen_paths)? {
                    return Ok(None);
                }

                conn.query_row(
                        "INSERT INTO files (path, kind, size, mtime, hash, state, width, height, folder)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, 0, ?7)
                         RETURNING id",
                        rusqlite::params![
                            path_to_text(path),
                            kind_to_i64(kind),
                            size as i64,
                            mtime,
                            hash.0.to_vec(),
                            state_to_i64(IndexState::Pending),
                            path_to_text(folder),
                        ],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(store_err)?
                    .ok_or_else(|| LumeError::Store("inserted file without returned id".into()))
                    .map(Some)
            }
        }
    }

    /// Delete indexed Items under `root` that are absent from the latest
    /// primary-Item walk. Hash-matched moves are updated before this pass, so
    /// their new path is present in `seen_paths` and their Units are preserved.
    pub fn delete_missing_files_under_root(
        &self,
        root: &Path,
        seen_paths: &BTreeSet<PathBuf>,
    ) -> Result<(), LumeError> {
        let mut conn = self.writer.lock().expect("writer lock poisoned");
        let rows = {
            let mut stmt = conn
                .prepare("SELECT id, path FROM files ORDER BY id")
                .map_err(store_err)?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, FileId>(0)?,
                        PathBuf::from(row.get::<_, String>(1)?),
                    ))
                })
                .map_err(store_err)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(store_err)?
        };

        let tx = conn.transaction().map_err(store_err)?;
        for (id, path) in rows {
            if path.starts_with(root) && !seen_paths.contains(&path) {
                delete_units_for_file(&tx, id)?;
                tx.execute("DELETE FROM files WHERE id = ?1", [id])
                    .map_err(store_err)?;
            }
        }
        tx.commit().map_err(store_err)
    }

    /// All `files` rows, oldest-inserted first.
    pub fn list_files(&self) -> Result<Vec<FileRecord>, LumeError> {
        let conn = self.writer.lock().expect("writer lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, path, kind, size, mtime, hash, state, captured_at,
                        width, height, duration_s, folder, gps_lat, gps_lng
                 FROM files ORDER BY id",
            )
            .map_err(store_err)?;
        let rows = stmt.query_map([], row_to_file_record).map_err(store_err)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(store_err)
    }

    /// Update a file's [`IndexState`]. `hash` overwrites the stored hash only
    /// when `Some` (M1 always passes `None` — eager hashing is M2).
    pub fn set_file_state(
        &self,
        file: FileId,
        state: IndexState,
        hash: Option<Blake3Hash>,
    ) -> Result<(), LumeError> {
        let conn = self.writer.lock().expect("writer lock poisoned");
        let changed = conn
            .execute(
                "UPDATE files SET state = ?1, hash = COALESCE(?2, hash) WHERE id = ?3",
                rusqlite::params![state_to_i64(state), hash.map(|h| h.0.to_vec()), file],
            )
            .map_err(store_err)?;
        if changed == 0 {
            return Err(LumeError::NotFound(format!("file {file}")));
        }
        Ok(())
    }

    /// Clear every `files`/`units`/`vec_units` row — M1's "full re-index
    /// every run" (incremental indexing is M2).
    pub fn reset_all(&self) -> Result<(), LumeError> {
        let conn = self.writer.lock().expect("writer lock poisoned");
        conn.execute_batch("DELETE FROM vec_units; DELETE FROM units; DELETE FROM files;")
            .map_err(store_err)?;
        Ok(())
    }
}

impl VectorStore for SqliteStore {
    /// Atomically insert one committed indexing batch: a `units` row per
    /// [`EmbeddedUnit`], keyed 1:1 to its `vec_units` row by rowid, all in one
    /// transaction (DESIGN §10's crash-resume commit boundary).
    fn insert_batch(&self, units: &[EmbeddedUnit<'_>]) -> Result<(), LumeError> {
        let mut conn = self.writer.lock().expect("writer lock poisoned");
        let tx = conn.transaction().map_err(store_err)?;
        {
            let mut insert_unit = tx
                .prepare("INSERT INTO units (file_id, frame_ts) VALUES (?1, ?2)")
                .map_err(store_err)?;
            let mut insert_vec = tx
                .prepare("INSERT INTO vec_units (rowid, embedding) VALUES (?1, ?2)")
                .map_err(store_err)?;
            for unit in units {
                insert_unit
                    .execute(rusqlite::params![unit.file, unit.frame_ts])
                    .map_err(store_err)?;
                let unit_id = tx.last_insert_rowid();
                insert_vec
                    .execute(rusqlite::params![unit_id, embedding_to_blob(unit.emb)])
                    .map_err(store_err)?;
            }
        }
        tx.commit().map_err(store_err)
    }

    /// Exact KNN over `vec_units`, with [`SearchFilters`] pushed into the same
    /// query as a `JOIN` against `files` (ADR-0003 — verified live) rather
    /// than filtering an already-fetched top-k in Rust.
    fn knn(
        &self,
        query: &Embedding,
        k: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<ScoredHit>, LumeError> {
        let conn = self.reader()?;
        let mut stmt = conn
            .prepare(
                "SELECT u.file_id, u.frame_ts, v.distance
                 FROM vec_units v
                 JOIN units u ON u.id = v.rowid
                 JOIN files f ON f.id = u.file_id
                 WHERE v.embedding MATCH ?1 AND k = ?2
                   AND (?3 IS NULL OR f.kind = ?3)
                   AND (?4 IS NULL OR f.captured_at >= ?4)
                   AND (?5 IS NULL OR f.captured_at <= ?5)
                   AND (?6 IS NULL OR f.folder = ?6)
                 ORDER BY v.distance",
            )
            .map_err(store_err)?;

        let kind_filter = filters.kind.map(kind_to_i64);
        let folder_filter = filters.folder.as_deref().map(path_to_text);

        let rows = stmt
            .query_map(
                rusqlite::params![
                    embedding_to_blob(query),
                    k as i64,
                    kind_filter,
                    filters.captured_after,
                    filters.captured_before,
                    folder_filter,
                ],
                |row| {
                    let distance: f32 = row.get(2)?;
                    Ok(ScoredHit {
                        file: row.get(0)?,
                        frame_ts: row.get(1)?,
                        score: 1.0 - (distance * distance) / 2.0,
                    })
                },
            )
            .map_err(store_err)?;

        rows.collect::<Result<Vec<_>, _>>().map_err(store_err)
    }

    /// Remove every `units`/`vec_units` row for `file`, then the `files` row
    /// itself, in one transaction.
    fn delete_file(&self, file: FileId) -> Result<(), LumeError> {
        let mut conn = self.writer.lock().expect("writer lock poisoned");
        let tx = conn.transaction().map_err(store_err)?;
        delete_units_for_file(&tx, file)?;
        tx.execute("DELETE FROM files WHERE id = ?1", [file])
            .map_err(store_err)?;
        tx.commit().map_err(store_err)
    }
}

fn open_writer(db_path: &Path) -> Result<Connection, LumeError> {
    register_sqlite_vec();
    let conn = Connection::open(db_path).map_err(store_err)?;
    configure_connection(&conn)?;
    schema::apply(&conn)?;
    Ok(conn)
}

fn open_reader(db_path: &Path) -> Result<Connection, LumeError> {
    register_sqlite_vec();
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(store_err)?;
    conn.busy_timeout(Duration::from_millis(5_000))
        .map_err(store_err)?;
    Ok(conn)
}

fn move_done_file_by_hash(
    conn: &Connection,
    path: &Path,
    kind: MediaKind,
    size: u64,
    mtime: i64,
    hash: Blake3Hash,
    seen_paths: &BTreeSet<PathBuf>,
) -> Result<bool, LumeError> {
    let folder = path.parent().unwrap_or_else(|| Path::new(""));
    let mut stmt = conn
        .prepare(
            "SELECT id, path FROM files
             WHERE hash = ?1 AND state = ?2 AND path != ?3
             ORDER BY id",
        )
        .map_err(store_err)?;
    let matches = stmt
        .query_map(
            rusqlite::params![
                hash.0.to_vec(),
                state_to_i64(IndexState::Done),
                path_to_text(path),
            ],
            |row| {
                Ok((
                    row.get::<_, FileId>(0)?,
                    PathBuf::from(row.get::<_, String>(1)?),
                ))
            },
        )
        .map_err(store_err)?;

    for row in matches {
        let (id, old_path) = row.map_err(store_err)?;
        if seen_paths.contains(&old_path) {
            continue;
        }
        conn.execute(
            "UPDATE files
             SET path = ?1, kind = ?2, size = ?3, mtime = ?4, hash = ?5, folder = ?6
             WHERE id = ?7",
            rusqlite::params![
                path_to_text(path),
                kind_to_i64(kind),
                size as i64,
                mtime,
                hash.0.to_vec(),
                path_to_text(folder),
                id,
            ],
        )
        .map_err(store_err)?;
        return Ok(true);
    }
    Ok(false)
}

fn delete_units_for_file(conn: &Connection, file: FileId) -> Result<(), LumeError> {
    conn.execute(
        "DELETE FROM vec_units WHERE rowid IN (SELECT id FROM units WHERE file_id = ?1)",
        [file],
    )
    .map_err(store_err)?;
    conn.execute("DELETE FROM units WHERE file_id = ?1", [file])
        .map_err(store_err)?;
    Ok(())
}

/// WAL + `busy_timeout` — writers don't block readers, and lock contention
/// retries instead of surfacing `SQLITE_BUSY` (BUILD.md §10 schema notes).
fn configure_connection(conn: &Connection) -> Result<(), LumeError> {
    conn.pragma_update_and_check(None, "journal_mode", "WAL", |_| Ok(()))
        .map_err(store_err)?;
    conn.busy_timeout(Duration::from_millis(5_000))
        .map_err(store_err)?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .map_err(store_err)?;
    Ok(())
}

fn row_to_file_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<FileRecord> {
    let hash: Option<Vec<u8>> = row.get(5)?;
    let gps_lat: Option<f64> = row.get(12)?;
    let gps_lng: Option<f64> = row.get(13)?;

    Ok(FileRecord {
        id: row.get(0)?,
        path: PathBuf::from(row.get::<_, String>(1)?),
        kind: i64_to_kind(row.get(2)?),
        size: row.get::<_, i64>(3)? as u64,
        mtime: row.get(4)?,
        hash: hash.map(|bytes| Blake3Hash(bytes_to_hash_array(&bytes))),
        state: i64_to_state(row.get(6)?),
        captured_at: row.get(7)?,
        width: row.get::<_, i64>(8)? as u32,
        height: row.get::<_, i64>(9)? as u32,
        duration_s: row.get(10)?,
        folder: PathBuf::from(row.get::<_, String>(11)?),
        gps: gps_lat.zip(gps_lng),
    })
}

fn bytes_to_hash_array(bytes: &[u8]) -> [u8; 32] {
    let mut arr = [0_u8; 32];
    let n = bytes.len().min(32);
    arr[..n].copy_from_slice(&bytes[..n]);
    arr
}

fn embedding_to_blob(emb: &Embedding) -> Vec<u8> {
    emb.0
        .iter()
        .flat_map(|v| v.to_f32().to_le_bytes())
        .collect()
}

fn path_to_text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn kind_to_i64(kind: MediaKind) -> i64 {
    match kind {
        MediaKind::Image => 0,
        MediaKind::Video => 1,
    }
}

fn i64_to_kind(v: i64) -> MediaKind {
    match v {
        1 => MediaKind::Video,
        _ => MediaKind::Image,
    }
}

fn state_to_i64(state: IndexState) -> i64 {
    match state {
        IndexState::Pending => 0,
        IndexState::Done => 1,
        IndexState::Failed => 2,
        IndexState::Stale => 3,
    }
}

fn i64_to_state(v: i64) -> IndexState {
    match v {
        1 => IndexState::Done,
        2 => IndexState::Failed,
        3 => IndexState::Stale,
        _ => IndexState::Pending,
    }
}

fn store_err(e: rusqlite::Error) -> LumeError {
    LumeError::Store(e.to_string())
}
