//! Pins the wire-contract *shape* (DESIGN §9 / BUILD.md L2) until the real
//! transport lands. A round-trip proves every message encodes and decodes
//! losslessly — including the per-unit `Failed` arm, which must survive so a
//! corrupt photo is reported in-band, not as a transport error (DESIGN §17).

use lume_ipc::protocol::{
    BatchItem, EmbedOneRequest, EmbedOneResponse, EmbedRequest, EmbedResponse, RequestUnit,
    UnitResult,
};

#[test]
fn embed_request_round_trips() {
    let req = EmbedRequest {
        batch_id: 7,
        units: vec![
            RequestUnit {
                unit_idx: 0,
                path: "/a/photo.jpg".into(),
                frame_ts: None,
            },
            RequestUnit {
                unit_idx: 1,
                path: "/a/clip.mov".into(),
                frame_ts: Some(12.5),
            },
        ],
    };
    let back: EmbedRequest = serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
    assert_eq!(req, back);
}

#[test]
fn embed_response_preserves_ok_and_failed_arms() {
    let resp = EmbedResponse {
        batch_id: 7,
        items: vec![
            BatchItem {
                unit_idx: 0,
                result: UnitResult::Ok {
                    emb_fp16: vec![1, 2, 3, 4],
                    thumb_jpeg: vec![255, 216, 255],
                },
            },
            BatchItem {
                unit_idx: 1,
                result: UnitResult::Failed {
                    reason: "unsupported codec".into(),
                },
            },
        ],
    };
    let back: EmbedResponse = serde_json::from_str(&serde_json::to_string(&resp).unwrap()).unwrap();
    assert_eq!(resp, back);
}

#[test]
fn embed_one_round_trips() {
    let req = EmbedOneRequest {
        image_bytes: vec![0xFF, 0xD8, 0xFF],
    };
    let resp = EmbedOneResponse {
        emb_fp16: vec![9, 9],
    };
    assert_eq!(
        req,
        serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap()
    );
    assert_eq!(
        resp,
        serde_json::from_str::<EmbedOneResponse>(&serde_json::to_string(&resp).unwrap()).unwrap()
    );
}
