//! Pins the wire-contract *shape* (DESIGN §9 / BUILD.md L2) until the real
//! transport lands. A round-trip proves every message encodes and decodes
//! losslessly — including the per-unit `Failed` arm, which must survive so a
//! corrupt photo is reported in-band, not as a transport error (DESIGN §17).

use lume_ipc::protocol::{
    BatchItem, EmbedOneRequest, EmbedOneResponse, EmbedRequest, EmbedResponse, EmbedTextRequest,
    RequestUnit, UnitResult,
};

fn fixture(name: &str) -> &'static str {
    match name {
        "embed_request" => include_str!("../../../wire-fixtures/embed_request.json"),
        "embed_response_ok_failed" => {
            include_str!("../../../wire-fixtures/embed_response_ok_failed.json")
        }
        "embed_one_request" => include_str!("../../../wire-fixtures/embed_one_request.json"),
        "embed_one_response" => include_str!("../../../wire-fixtures/embed_one_response.json"),
        "embed_text_request" => include_str!("../../../wire-fixtures/embed_text_request.json"),
        _ => unreachable!("unknown fixture"),
    }
    .trim_end_matches('\n')
}

#[test]
fn embed_request_round_trips() {
    let req = EmbedRequest {
        batch_id: 7,
        thumb_px: 400,
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
fn embed_request_matches_shared_wire_fixture() {
    let req = EmbedRequest {
        batch_id: 7,
        thumb_px: 400,
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

    assert_eq!(
        serde_json::to_string(&req).unwrap(),
        fixture("embed_request")
    );
    assert_eq!(
        serde_json::from_str::<EmbedRequest>(fixture("embed_request")).unwrap(),
        req
    );
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
fn embed_response_matches_shared_wire_fixture() {
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

    assert_eq!(
        serde_json::to_string(&resp).unwrap(),
        fixture("embed_response_ok_failed")
    );
    assert_eq!(
        serde_json::from_str::<EmbedResponse>(fixture("embed_response_ok_failed")).unwrap(),
        resp
    );
}

#[test]
fn binary_fields_serialize_as_base64_strings_not_int_arrays() {
    // Transport depth (issue #17): emb_fp16 / thumb_jpeg / image_bytes must
    // cross the wire as compact base64 strings, never as JSON integer arrays.
    let ok = serde_json::to_string(&UnitResult::Ok {
        emb_fp16: vec![1, 2, 3, 4],
        thumb_jpeg: vec![0xFF, 0xD8, 0xFF],
    })
    .unwrap();
    assert!(
        ok.contains(r#""emb_fp16":"AQIDBA==""#),
        "emb_fp16 must be a base64 string, got {ok}"
    );
    assert!(
        ok.contains(r#""thumb_jpeg":"/9j/""#),
        "thumb_jpeg must be a base64 string, got {ok}"
    );

    let req = serde_json::to_string(&EmbedOneRequest {
        image_bytes: vec![0xFF, 0xD8, 0xFF],
    })
    .unwrap();
    assert_eq!(req, r#"{"image_bytes":"/9j/"}"#);
    assert!(
        !req.contains('['),
        "image_bytes must not serialize as an int array, got {req}"
    );

    let resp = serde_json::to_string(&EmbedOneResponse {
        emb_fp16: vec![9, 9],
    })
    .unwrap();
    assert_eq!(resp, r#"{"emb_fp16":"CQk="}"#);
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

#[test]
fn embed_one_messages_match_shared_wire_fixtures() {
    let req = EmbedOneRequest {
        image_bytes: vec![0xFF, 0xD8, 0xFF],
    };
    let resp = EmbedOneResponse {
        emb_fp16: vec![9, 9],
    };

    assert_eq!(
        serde_json::to_string(&req).unwrap(),
        fixture("embed_one_request")
    );
    assert_eq!(
        serde_json::from_str::<EmbedOneRequest>(fixture("embed_one_request")).unwrap(),
        req
    );
    assert_eq!(
        serde_json::to_string(&resp).unwrap(),
        fixture("embed_one_response")
    );
    assert_eq!(
        serde_json::from_str::<EmbedOneResponse>(fixture("embed_one_response")).unwrap(),
        resp
    );
}

#[test]
fn embed_text_request_uses_query_text_only() {
    let req = EmbedTextRequest {
        text: "girl riding a bicycle".into(),
    };
    let resp = EmbedOneResponse {
        emb_fp16: vec![8, 8],
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

#[test]
fn embed_text_request_matches_shared_wire_fixture() {
    let req = EmbedTextRequest {
        text: "girl riding a bicycle".into(),
    };

    assert_eq!(
        serde_json::to_string(&req).unwrap(),
        fixture("embed_text_request")
    );
    assert_eq!(
        serde_json::from_str::<EmbedTextRequest>(fixture("embed_text_request")).unwrap(),
        req
    );
}
