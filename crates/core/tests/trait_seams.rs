use std::cell::RefCell;
use std::path::PathBuf;

use lume_core::{
    EmbedOutcome, EmbedUnit, EmbeddedUnit, Embedding, FileId, LumeError, MediaKind, ScoredHit,
    SearchFilters, Sidecar, VectorStore,
};

#[derive(Default)]
struct RecordingStore {
    inserted: RefCell<Vec<(FileId, Option<f32>, usize)>>,
    last_filters: RefCell<Option<SearchFilters>>,
}

impl VectorStore for RecordingStore {
    fn insert_batch(&self, units: &[EmbeddedUnit<'_>]) -> Result<(), LumeError> {
        self.inserted
            .borrow_mut()
            .extend(units.iter().map(|u| (u.file, u.frame_ts, u.emb.dim())));
        Ok(())
    }

    fn knn(
        &self,
        _query: &Embedding,
        _k: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<ScoredHit>, LumeError> {
        self.last_filters.replace(Some(filters.clone()));
        Ok(Vec::new())
    }

    fn delete_file(&self, _file: FileId) -> Result<(), LumeError> {
        Ok(())
    }
}

#[test]
fn vector_store_batches_units_and_receives_search_filters() {
    let store = RecordingStore::default();
    let emb = Embedding(Vec::new());
    let batch = [
        EmbeddedUnit {
            file: 10,
            frame_ts: None,
            emb: &emb,
        },
        EmbeddedUnit {
            file: 11,
            frame_ts: Some(4.5),
            emb: &emb,
        },
    ];

    store.insert_batch(&batch).unwrap();

    let filters = SearchFilters {
        kind: Some(MediaKind::Image),
        captured_after: Some(1_700_000_000),
        captured_before: Some(1_800_000_000),
        folder: Some(PathBuf::from("/library/trips")),
    };
    store.knn(&emb, 500, &filters).unwrap();

    assert_eq!(
        *store.inserted.borrow(),
        vec![(10, None, 0), (11, Some(4.5), 0)]
    );
    assert_eq!(*store.last_filters.borrow(), Some(filters));
}

struct TextCapableSidecar;

impl Sidecar for TextCapableSidecar {
    fn embed(&self, units: &[EmbedUnit]) -> Result<Vec<EmbedOutcome>, LumeError> {
        Ok(units
            .iter()
            .map(|_| EmbedOutcome::Ok {
                emb: Embedding(Vec::new()),
                thumbnail_jpeg: Vec::new(),
            })
            .collect())
    }

    fn embed_one(&self, _image: &[u8]) -> Result<Embedding, LumeError> {
        Ok(Embedding(Vec::new()))
    }

    fn embed_text(&self, query: &str) -> Result<Embedding, LumeError> {
        Ok(Embedding(vec![half::f16::from_f32(query.len() as f32)]))
    }
}

#[test]
fn sidecar_exposes_text_query_embedding() {
    let emb = TextCapableSidecar
        .embed_text("girl riding a bicycle")
        .unwrap();
    assert_eq!(emb.dim(), 1);
}
