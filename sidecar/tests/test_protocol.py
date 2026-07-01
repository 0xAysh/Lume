"""Round-trip tests pinning the wire-contract shape (mirrors the Rust
`wire_shape` test). Verifies the `UnitResult` union keeps both arms — the
in-band `Failed` arm must survive so a corrupt photo is reported, not dropped
(DESIGN §17)."""

import json
from pathlib import Path

from lume_sidecar.protocol import (
    BatchItem,
    EmbedOneRequest,
    EmbedOneResponse,
    EmbedRequest,
    EmbedResponse,
    EmbedTextRequest,
    RequestUnit,
    UnitFailed,
    UnitOk,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "wire-fixtures"


def _json_round_trip(obj_dict: dict) -> dict:
    return json.loads(json.dumps(obj_dict))


def _compact_json(obj_dict: dict) -> str:
    return json.dumps(obj_dict, separators=(",", ":"))


def _fixture(name: str) -> str:
    return (FIXTURE_DIR / f"{name}.json").read_text().rstrip("\n")


def test_embed_request_round_trips():
    req = EmbedRequest(
        batch_id=7,
        thumb_px=400,
        units=[
            RequestUnit(unit_idx=0, path="/a/photo.jpg"),
            RequestUnit(unit_idx=1, path="/a/clip.mov", frame_ts=12.5),
        ],
    )
    back = EmbedRequest.from_dict(_json_round_trip(req.to_dict()))
    assert back == req


def test_embed_request_matches_shared_wire_fixture():
    req = EmbedRequest(
        batch_id=7,
        thumb_px=400,
        units=[
            RequestUnit(unit_idx=0, path="/a/photo.jpg"),
            RequestUnit(unit_idx=1, path="/a/clip.mov", frame_ts=12.5),
        ],
    )

    assert _compact_json(req.to_dict()) == _fixture("embed_request")
    assert EmbedRequest.from_dict(json.loads(_fixture("embed_request"))) == req


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


def test_embed_response_matches_shared_wire_fixture():
    resp = EmbedResponse(
        batch_id=7,
        items=[
            BatchItem(
                unit_idx=0,
                result=UnitOk(emb_fp16=b"\x01\x02\x03\x04", thumb_jpeg=b"\xff\xd8\xff"),
            ),
            BatchItem(unit_idx=1, result=UnitFailed(reason="unsupported codec")),
        ],
    )

    assert _compact_json(resp.to_dict()) == _fixture("embed_response_ok_failed")
    assert EmbedResponse.from_dict(json.loads(_fixture("embed_response_ok_failed"))) == resp


def test_embed_one_round_trips():
    req = EmbedOneRequest(image_bytes=b"\xff\xd8\xff")
    resp = EmbedOneResponse(emb_fp16=b"\x09\x09")

    assert EmbedOneRequest.from_dict(_json_round_trip(req.to_dict())) == req
    assert EmbedOneResponse.from_dict(_json_round_trip(resp.to_dict())) == resp


def test_embed_one_messages_match_shared_wire_fixtures():
    req = EmbedOneRequest(image_bytes=b"\xff\xd8\xff")
    resp = EmbedOneResponse(emb_fp16=b"\x09\x09")

    assert _compact_json(req.to_dict()) == _fixture("embed_one_request")
    assert EmbedOneRequest.from_dict(json.loads(_fixture("embed_one_request"))) == req
    assert _compact_json(resp.to_dict()) == _fixture("embed_one_response")
    assert EmbedOneResponse.from_dict(json.loads(_fixture("embed_one_response"))) == resp


def test_embed_text_request_round_trips():
    req = EmbedTextRequest(text="girl riding a bicycle")
    resp = EmbedOneResponse(emb_fp16=b"\x08\x08")

    assert EmbedTextRequest.from_dict(_json_round_trip(req.to_dict())) == req
    assert EmbedOneResponse.from_dict(_json_round_trip(resp.to_dict())) == resp


def test_embed_text_request_matches_shared_wire_fixture():
    req = EmbedTextRequest(text="girl riding a bicycle")

    assert _compact_json(req.to_dict()) == _fixture("embed_text_request")
    assert EmbedTextRequest.from_dict(json.loads(_fixture("embed_text_request"))) == req
