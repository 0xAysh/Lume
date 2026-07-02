//! Regression tests for the batch commit interface (issue #16): the one local
//! module that owns "an embedded Item becomes visible" across its Tile
//! thumbnail on disk, its Unit vector rows, and its `files.state`. These
//! exercise the public commit interface directly (not through `Indexer`) so a
//! fake failing `VectorStore` seam can be injected without touching the real
//! sidecar.

use std::fs;
use std::path::PathBuf;

use half::f16;
use lume_core::{
    Blake3Hash, EmbedOutcome, EmbeddedUnit, Embedding, FileId, IndexState, LumeError, MediaKind,
    ScoredHit, SearchFilters, VectorStore,
};
use lume_index::BatchCommitter;
use lume_store::SqliteStore;

const STUB_JPEG: &[u8] = &[0xFF, 0xD8, 0xFF, 0xD9];

fn ok_outcome() -> EmbedOutcome {
    EmbedOutcome::Ok {
        emb: Embedding(vec![f16::from_f32(1.0); Embedding::DIM]),
        thumbnail_jpeg: STUB_JPEG.to_vec(),
    }
}

fn setup() -> (tempfile::TempDir, SqliteStore, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let thumbs = dir.path().join("thumbnails");
    fs::create_dir_all(&thumbs).unwrap();
    let store = SqliteStore::open(dir.path().join("lume.sqlite3")).unwrap();
    (dir, store, thumbs)
}

fn state_of(store: &SqliteStore, id: FileId) -> IndexState {
    store
        .list_files()
        .unwrap()
        .into_iter()
        .find(|f| f.id == id)
        .unwrap()
        .state
}

fn hash(byte: u8) -> Blake3Hash {
    Blake3Hash([byte; 32])
}

/// A thumbnail write error for one Item must never leave that Item `Done`
/// without its Tile, and must never abort the sibling Units in the same batch.
#[test]
fn thumbnail_write_failure_does_not_mark_item_done_and_spares_siblings() {
    let (_dir, store, thumbs) = setup();
    let a = store
        .upsert_file(&PathBuf::from("/lib/a.jpg"), MediaKind::Image, 1, 1)
        .unwrap();
    let b = store
        .upsert_file(&PathBuf::from("/lib/b.jpg"), MediaKind::Image, 1, 1)
        .unwrap();

    // Plant a directory where a's `{id}.jpg` thumbnail must go, so the file
    // write for a — and only a — fails. b's thumbnail path stays writable.
    fs::create_dir(thumbs.join(format!("{a}.jpg"))).unwrap();

    let committer = BatchCommitter::new(&thumbs, &store, &store);
    committer
        .commit(
            &[a, b],
            &[hash(1), hash(2)],
            vec![ok_outcome(), ok_outcome()],
        )
        .unwrap();

    // a: its thumbnail write failed, so it must NOT be Done.
    assert_ne!(state_of(&store, a), IndexState::Done);
    assert_eq!(state_of(&store, a), IndexState::Failed);

    // b: committed fully despite a's failure (no sibling abort).
    assert_eq!(state_of(&store, b), IndexState::Done);
    assert_eq!(
        store
            .list_files()
            .unwrap()
            .into_iter()
            .find(|f| f.id == b)
            .unwrap()
            .hash,
        Some(hash(2))
    );
    assert_eq!(
        fs::read(thumbs.join(format!("{b}.jpg"))).unwrap(),
        STUB_JPEG
    );

    // Only b produced a searchable Unit.
    let q = Embedding(vec![f16::from_f32(1.0); Embedding::DIM]);
    let hits = store.knn(&q, 10, &SearchFilters::default()).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].file, b);
}

/// A `VectorStore` that always rejects the batch — proves that a vector-commit
/// failure never yields a `Done`-but-unsearchable Item, even though thumbnails
/// were already written to disk.
struct FailingVectors;

impl VectorStore for FailingVectors {
    fn insert_batch(&self, _units: &[EmbeddedUnit<'_>]) -> Result<(), LumeError> {
        Err(LumeError::Store("vector store is down".into()))
    }

    fn knn(
        &self,
        _query: &Embedding,
        _k: usize,
        _filters: &SearchFilters,
    ) -> Result<Vec<ScoredHit>, LumeError> {
        Ok(Vec::new())
    }

    fn delete_file(&self, _file: FileId) -> Result<(), LumeError> {
        Ok(())
    }
}

#[test]
fn vector_insert_failure_leaves_no_item_done_or_searchable() {
    let (_dir, store, thumbs) = setup();
    let a = store
        .upsert_file(&PathBuf::from("/lib/a.jpg"), MediaKind::Image, 1, 1)
        .unwrap();
    let b = store
        .upsert_file(&PathBuf::from("/lib/b.jpg"), MediaKind::Image, 1, 1)
        .unwrap();

    // Vector insert fails, but state transitions still go to the real store so
    // the test can observe that no Item was marked Done.
    let vectors = FailingVectors;
    let committer = BatchCommitter::new(&thumbs, &vectors, &store);
    let result = committer.commit(
        &[a, b],
        &[hash(1), hash(2)],
        vec![ok_outcome(), ok_outcome()],
    );

    // The commit surfaces the vector-store error rather than swallowing it.
    assert!(result.is_err());

    // No Item is Done, even though both thumbnails were written to disk.
    assert_ne!(state_of(&store, a), IndexState::Done);
    assert_ne!(state_of(&store, b), IndexState::Done);
    assert!(thumbs.join(format!("{a}.jpg")).exists());
    assert!(thumbs.join(format!("{b}.jpg")).exists());

    // Nothing is searchable in the real store: no Units were committed.
    let q = Embedding(vec![f16::from_f32(1.0); Embedding::DIM]);
    assert_eq!(
        store.knn(&q, 10, &SearchFilters::default()).unwrap().len(),
        0
    );
}
