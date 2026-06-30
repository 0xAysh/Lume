"""Round-trip tests pinning the wire-contract shape (mirrors the Rust
`wire_shape` test). Verifies the `UnitResult` union keeps both arms — the
in-band `Failed` arm must survive so a corrupt photo is reported, not dropped
(DESIGN §17)."""

import json

from lume_sidecar.protocol import (
    BatchItem,
    EmbedRequest,
    EmbedResponse,
    RequestUnit,
    UnitFailed,
    UnitOk,
)


def _json_round_trip(obj_dict: dict) -> dict:
    return json.loads(json.dumps(obj_dict))


def test_embed_request_round_trips():
    req = EmbedRequest(
        batch_id=7,
        units=[
            RequestUnit(unit_idx=0, path="/a/photo.jpg"),
            RequestUnit(unit_idx=1, path="/a/clip.mov", frame_ts=12.5),
        ],
    )
    back = EmbedRequest.from_dict(_json_round_trip(req.to_dict()))
    assert back == req


def test_request_unit_omits_frame_ts_for_images():
    # Matches the Rust `skip_serializing_if = "Option::is_none"` so an image Unit
    # and a video-frame Unit are wire-distinguishable.
    assert "frame_ts" not in RequestUnit(unit_idx=0, path="/a.jpg").to_dict()
    assert RequestUnit(unit_idx=0, path="/a.mov", frame_ts=1.0).to_dict()["frame_ts"] == 1.0


def test_embed_response_preserves_ok_and_failed_arms():
    resp = EmbedResponse(
        batch_id=7,
        items=[
            BatchItem(unit_idx=0, result=UnitOk(emb_fp16=b"\x01\x02", thumb_jpeg=b"\xff\xd8")),
            BatchItem(unit_idx=1, result=UnitFailed(reason="unsupported codec")),
        ],
    )
    back = EmbedResponse.from_dict(_json_round_trip(resp.to_dict()))
    assert back == resp
    assert isinstance(back.items[0].result, UnitOk)
    assert isinstance(back.items[1].result, UnitFailed)
