//! Behavioural tests for [`Indexer`] against the real [`SqliteStore`] (M1
//! Slice 1) and a fake [`Sidecar`] — the acceptance criteria only require the
//! real embedder for the *manual* end-to-end check (M1 Slice 2's own scope);
//! Rust-level tests exercise the Indexer's own contract (batching, per-file
//! failure isolation, idempotent re-run, observable progress) against a fake.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use half::f16;
use lume_core::{EmbedOutcome, EmbedUnit, Embedding, IndexState, LumeError, Sidecar, VectorStore};
use lume_index::Indexer;
use lume_store::SqliteStore;

/// A [`Sidecar`] that never touches the disk: it fails units whose filename
/// is in `fail_names`, and otherwise returns a deterministic unit-length
/// embedding + a minimal-but-valid JPEG stub. `embed_delay` lets the
/// progress-observability test catch the Indexer mid-batch.
struct FakeSidecar {
    fail_names: Vec<String>,
    embed_calls: AtomicUsize,
    embed_delay: Duration,
}

impl FakeSidecar {
    fn new(fail_names: Vec<&str>) -> Self {
        Self {
            fail_names: fail_names.into_iter().map(String::from).collect(),
            embed_calls: AtomicUsize::new(0),
            embed_delay: Duration::ZERO,
        }
    }

    fn with_delay(mut self, delay: Duration) -> Self {
        self.embed_delay = delay;
        self
    }
}

const STUB_JPEG: &[u8] = &[0xFF, 0xD8, 0xFF, 0xD9];

impl Sidecar for FakeSidecar {
    fn embed(&self, units: &[EmbedUnit]) -> Result<Vec<EmbedOutcome>, LumeError> {
        self.embed_calls.fetch_add(1, Ordering::SeqCst);
        if !self.embed_delay.is_zero() {
            std::thread::sleep(self.embed_delay);
        }
        Ok(units
            .iter()
            .map(|u| {
                let name = u.path.file_name().unwrap().to_string_lossy().into_owned();
                if self.fail_names.contains(&name) {
                    EmbedOutcome::Failed {
                        reason: "corrupt test fixture".into(),
                    }
                } else {
                    EmbedOutcome::Ok {
                        emb: Embedding(vec![f16::from_f32(1.0); Embedding::DIM]),
                        thumbnail_jpeg: STUB_JPEG.to_vec(),
                    }
                }
            })
            .collect())
    }

    fn embed_one(&self, _image: &[u8]) -> Result<Embedding, LumeError> {
        unimplemented!("not exercised by Indexer")
    }

    fn embed_text(&self, _query: &str) -> Result<Embedding, LumeError> {
        unimplemented!("not exercised by Indexer")
    }
}

fn setup(fail_names: Vec<&str>) -> (tempfile::TempDir, Arc<SqliteStore>, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("library");
    fs::create_dir_all(&root).unwrap();
    let thumbs = dir.path().join("thumbnails");
    let store = Arc::new(SqliteStore::open(dir.path().join("lume.sqlite3")).unwrap());
    let _ = fail_names;
    (dir, store, root, thumbs)
}

#[test]
fn indexes_every_valid_file_with_thumbnail_and_done_state() {
    let (_dir, store, root, thumbs) = setup(vec![]);
    fs::write(root.join("a.jpg"), b"fixture").unwrap();
    fs::write(root.join("b.png"), b"fixture").unwrap();
    fs::write(root.join("ignored.txt"), b"fixture").unwrap();

    let sidecar = Arc::new(FakeSidecar::new(vec![]));
    let indexer = Indexer::new(root, 32, thumbs.clone(), Arc::clone(&store), sidecar);
    indexer.run().unwrap();

    let files = store.list_files().unwrap();
    assert_eq!(files.len(), 2, "only jpg/png are indexed, txt is ignored");
    for file in &files {
        assert_eq!(file.state, IndexState::Done);
        let thumb_path = thumbs.join(format!("{}.jpg", file.id));
        assert_eq!(fs::read(&thumb_path).unwrap(), STUB_JPEG);
    }

    // Every Done file produced a searchable Unit/vec row.
    let query = Embedding(vec![f16::from_f32(1.0); Embedding::DIM]);
    let hits = store
        .knn(&query, 10, &lume_core::SearchFilters::default())
        .unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn a_corrupt_file_fails_without_aborting_its_batch() {
    let (_dir, store, root, thumbs) = setup(vec![]);
    fs::write(root.join("good_a.jpg"), b"fixture").unwrap();
    fs::write(root.join("corrupt.jpg"), b"fixture").unwrap();
    fs::write(root.join("good_b.jpg"), b"fixture").unwrap();

    // batch_size=32 keeps all three files in one batch, proving a single
    // in-band failure doesn't abort sibling Units in the same batch.
    let sidecar = Arc::new(FakeSidecar::new(vec!["corrupt.jpg"]));
    let indexer = Indexer::new(root, 32, thumbs, Arc::clone(&store), sidecar);
    indexer.run().unwrap();

    let files = store.list_files().unwrap();
    assert_eq!(files.len(), 3);

    let by_name = |name: &str| {
        files
            .iter()
            .find(|f| f.path.file_name().unwrap() == name)
            .unwrap()
            .state
    };
    assert_eq!(by_name("good_a.jpg"), IndexState::Done);
    assert_eq!(by_name("good_b.jpg"), IndexState::Done);
    assert_eq!(by_name("corrupt.jpg"), IndexState::Failed);
}

#[test]
fn rerunning_the_indexer_does_not_duplicate_rows() {
    let (_dir, store, root, thumbs) = setup(vec![]);
    fs::write(root.join("a.jpg"), b"fixture").unwrap();
    fs::write(root.join("b.jpg"), b"fixture").unwrap();

    let indexer = Indexer::new(
        root,
        32,
        thumbs,
        Arc::clone(&store),
        Arc::new(FakeSidecar::new(vec![])),
    );
    indexer.run().unwrap();
    indexer.run().unwrap();

    assert_eq!(store.list_files().unwrap().len(), 2);
    let query = Embedding(vec![f16::from_f32(1.0); Embedding::DIM]);
    let hits = store
        .knn(&query, 10, &lume_core::SearchFilters::default())
        .unwrap();
    assert_eq!(hits.len(), 2, "re-index must not duplicate Units either");
}

#[test]
fn progress_is_observable_while_the_run_is_in_flight() {
    let (_dir, store, root, thumbs) = setup(vec![]);
    for i in 0..6 {
        fs::write(root.join(format!("{i}.jpg")), b"fixture").unwrap();
    }

    let sidecar = Arc::new(FakeSidecar::new(vec![]).with_delay(Duration::from_millis(40)));
    // batch_size=1 forces 6 separate `embed` calls, each pausing 40ms, so
    // there's a wide window in which a concurrent poller must see a strictly
    // partial (done < total) snapshot.
    let indexer = Arc::new(Indexer::new(root, 1, thumbs, Arc::clone(&store), sidecar));
    let progress = indexer.progress();

    let run_handle = {
        let indexer = Arc::clone(&indexer);
        std::thread::spawn(move || indexer.run().unwrap())
    };

    let mut saw_partial_progress = false;
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !run_handle.is_finished() && std::time::Instant::now() < deadline {
        let (done, total) = progress.snapshot();
        if total > 0 && done > 0 && done < total {
            saw_partial_progress = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    run_handle.join().unwrap();

    assert!(
        saw_partial_progress,
        "expected to observe a partial (done < total) progress snapshot mid-run"
    );
    assert_eq!(progress.snapshot(), (6, 6));
}
