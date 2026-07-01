//! Behavioural tests for [`SqliteStore`] through its public interface only
//! (`VectorStore` + the metadata helpers) — no reaching into SQL internals.

use std::path::PathBuf;

use half::f16;
use lume_core::{
    Dtype, EmbeddedUnit, Embedding, IndexState, MediaKind, SearchFilters, VectorStore,
};
use lume_store::SqliteStore;

fn temp_store() -> (tempfile::TempDir, SqliteStore) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("lume.sqlite3");
    let store = SqliteStore::open(&db_path).expect("open store");
    (dir, store)
}

/// Zero-pad to `Embedding::DIM` — trailing zeros don't affect dot product or
/// L2 norm, so short hand-written vectors stay meaningful for cosine tests.
fn emb(prefix: &[f32]) -> Embedding {
    let mut v = vec![f16::from_f32(0.0); Embedding::DIM];
    for (slot, val) in v.iter_mut().zip(prefix) {
        *slot = f16::from_f32(*val);
    }
    Embedding(v)
}

fn normalize(prefix: &[f32]) -> Vec<f32> {
    let norm = prefix.iter().map(|v| v * v).sum::<f32>().sqrt();
    prefix.iter().map(|v| v / norm).collect()
}

#[test]
fn default_config_stores_vectors_as_f32_per_adr_0003() {
    assert_eq!(lume_core::Config::default().vector_dtype, Dtype::F32);
}

#[test]
fn insert_batch_and_knn_round_trip_ordered_by_similarity() {
    let (_dir, store) = temp_store();

    let file_a = store
        .upsert_file(
            &PathBuf::from("/lib/a.jpg"),
            MediaKind::Image,
            100,
            1_700_000_000,
        )
        .unwrap();
    let file_b = store
        .upsert_file(
            &PathBuf::from("/lib/b.jpg"),
            MediaKind::Image,
            200,
            1_700_000_100,
        )
        .unwrap();
    let file_c = store
        .upsert_file(
            &PathBuf::from("/lib/c.jpg"),
            MediaKind::Image,
            300,
            1_700_000_200,
        )
        .unwrap();

    // Hand-computed cosine similarity: query = (1,0); a = (1,0) -> cos=1.0;
    // b = normalize(1,1) -> cos=0.707; c = (0,1) -> cos=0.0. Expected order:
    // a, b, c.
    let query = emb(&normalize(&[1.0, 0.0]));
    let emb_a = emb(&normalize(&[1.0, 0.0]));
    let emb_b = emb(&normalize(&[1.0, 1.0]));
    let emb_c = emb(&normalize(&[0.0, 1.0]));

    store
        .insert_batch(&[
            EmbeddedUnit {
                file: file_c,
                frame_ts: None,
                emb: &emb_c,
            },
            EmbeddedUnit {
                file: file_a,
                frame_ts: None,
                emb: &emb_a,
            },
            EmbeddedUnit {
                file: file_b,
                frame_ts: None,
                emb: &emb_b,
            },
        ])
        .unwrap();

    let hits = store.knn(&query, 10, &SearchFilters::default()).unwrap();
    let order: Vec<_> = hits.iter().map(|h| h.file).collect();
    assert_eq!(order, vec![file_a, file_b, file_c]);

    // Cosine score recovery from L2 distance (ADR-0003): a is an exact match.
    assert!((hits[0].score - 1.0).abs() < 1e-3, "got {}", hits[0].score);
    assert!((hits[1].score - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-2);
    assert!(hits[2].score.abs() < 1e-2);
}

#[test]
fn knn_pushes_filters_into_the_store_query() {
    let (_dir, store) = temp_store();

    let image = store
        .upsert_file(&PathBuf::from("/lib/photo.jpg"), MediaKind::Image, 1, 1)
        .unwrap();
    let video = store
        .upsert_file(&PathBuf::from("/lib/clip.mov"), MediaKind::Video, 1, 1)
        .unwrap();

    let v = emb(&[1.0]);
    store
        .insert_batch(&[
            EmbeddedUnit {
                file: image,
                frame_ts: None,
                emb: &v,
            },
            EmbeddedUnit {
                file: video,
                frame_ts: Some(1.5),
                emb: &v,
            },
        ])
        .unwrap();

    let filters = SearchFilters {
        kind: Some(MediaKind::Video),
        ..Default::default()
    };
    let hits = store.knn(&v, 10, &filters).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].file, video);
}

#[test]
fn delete_file_removes_its_units_and_vectors() {
    let (_dir, store) = temp_store();
    let file = store
        .upsert_file(&PathBuf::from("/lib/a.jpg"), MediaKind::Image, 1, 1)
        .unwrap();
    let v = emb(&[1.0]);
    store
        .insert_batch(&[EmbeddedUnit {
            file,
            frame_ts: None,
            emb: &v,
        }])
        .unwrap();
    assert_eq!(
        store.knn(&v, 10, &SearchFilters::default()).unwrap().len(),
        1
    );

    store.delete_file(file).unwrap();

    assert_eq!(
        store.knn(&v, 10, &SearchFilters::default()).unwrap().len(),
        0
    );
    assert!(store.list_files().unwrap().is_empty());
}

#[test]
fn set_file_state_updates_state_and_optional_hash() {
    let (_dir, store) = temp_store();
    let file = store
        .upsert_file(&PathBuf::from("/lib/a.jpg"), MediaKind::Image, 1, 1)
        .unwrap();

    store.set_file_state(file, IndexState::Done, None).unwrap();
    let record = store.list_files().unwrap().into_iter().next().unwrap();
    assert_eq!(record.state, IndexState::Done);
    assert!(record.hash.is_none());

    let hash = lume_core::Blake3Hash([7_u8; 32]);
    store
        .set_file_state(file, IndexState::Failed, Some(hash))
        .unwrap();
    let record = store.list_files().unwrap().into_iter().next().unwrap();
    assert_eq!(record.state, IndexState::Failed);
    assert_eq!(record.hash, Some(hash));
}

#[test]
fn reset_all_clears_files_units_and_vectors() {
    let (_dir, store) = temp_store();
    let file = store
        .upsert_file(&PathBuf::from("/lib/a.jpg"), MediaKind::Image, 1, 1)
        .unwrap();
    let v = emb(&[1.0]);
    store
        .insert_batch(&[EmbeddedUnit {
            file,
            frame_ts: None,
            emb: &v,
        }])
        .unwrap();

    store.reset_all().unwrap();

    assert!(store.list_files().unwrap().is_empty());
    assert_eq!(
        store.knn(&v, 10, &SearchFilters::default()).unwrap().len(),
        0
    );
}

#[test]
fn upsert_file_is_idempotent_by_path() {
    let (_dir, store) = temp_store();
    let path = PathBuf::from("/lib/a.jpg");
    let first = store.upsert_file(&path, MediaKind::Image, 1, 1).unwrap();
    let second = store.upsert_file(&path, MediaKind::Image, 2, 2).unwrap();
    assert_eq!(first, second);
    assert_eq!(store.list_files().unwrap().len(), 1);
}

/// M1's real exit criterion (BUILD.md, PRD #1): a search returns correct
/// ranked results *while* a background index is actively committing batches.
/// Proves WAL + sqlite-vec are correct under concurrent append, not just
/// "some order" from a single-threaded test.
#[test]
fn concurrent_insert_batch_and_knn_never_error() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(SqliteStore::open(dir.path().join("lume.sqlite3")).unwrap());

    const FILE_COUNT: i64 = 400;
    let file_ids: Vec<_> = (0..FILE_COUNT)
        .map(|i| {
            store
                .upsert_file(
                    &PathBuf::from(format!("/lib/{i}.jpg")),
                    MediaKind::Image,
                    1,
                    i,
                )
                .unwrap()
        })
        .collect();

    let writer_store = Arc::clone(&store);
    let writer_errors = Arc::new(AtomicUsize::new(0));
    let writer_errors_thread = Arc::clone(&writer_errors);
    let writer = std::thread::spawn(move || {
        let v = emb(&[1.0]);
        for &file in &file_ids {
            let batch = [EmbeddedUnit {
                file,
                frame_ts: None,
                emb: &v,
            }];
            if writer_store.insert_batch(&batch).is_err() {
                writer_errors_thread.fetch_add(1, Ordering::SeqCst);
            }
        }
    });

    let mut observed_max_rows = 0_usize;
    let mut reader_errors = 0_usize;
    let v = emb(&[1.0]);
    while !writer.is_finished() {
        match store.knn(&v, FILE_COUNT as usize, &SearchFilters::default()) {
            Ok(hits) => observed_max_rows = observed_max_rows.max(hits.len()),
            Err(_) => reader_errors += 1,
        }
    }
    // Drain a final read after the writer has finished so we see the fully
    // committed state, not just whatever was visible mid-race.
    let final_hits = store
        .knn(&v, FILE_COUNT as usize, &SearchFilters::default())
        .unwrap();
    observed_max_rows = observed_max_rows.max(final_hits.len());

    writer.join().unwrap();

    assert_eq!(reader_errors, 0, "knn saw errors during concurrent writes");
    assert_eq!(
        writer_errors.load(Ordering::SeqCst),
        0,
        "insert_batch saw errors during concurrent reads"
    );
    assert_eq!(
        final_hits.len(),
        FILE_COUNT as usize,
        "all committed rows must be visible after the writer finishes"
    );
    assert!(
        observed_max_rows > 0,
        "the reader thread should have observed at least some progressively committed rows"
    );
}
