import socket

from lume_sidecar.framing import read_frame, write_frame
from lume_sidecar.protocol import EmbedOneResponse
from lume_sidecar.server import handle_message


def test_length_prefixed_json_frame_round_trips():
    left, right = socket.socketpair()
    try:
        write_frame(left, {"type": "embed_one_response", "payload": {"emb_fp16": [0, 60]}})

        assert read_frame(right) == {
            "type": "embed_one_response",
            "payload": {"emb_fp16": [0, 60]},
        }
    finally:
        left.close()
        right.close()


def test_fake_sidecar_handles_text_embed_message():
    response = handle_message({"type": "embed_text", "payload": {"text": "girl riding a bicycle"}})

    assert response == {
        "type": "embed_one_response",
        "payload": EmbedOneResponse(emb_fp16=bytes(1536)).to_dict(),
    }


def test_fake_sidecar_handles_batch_and_preserves_unit_indices():
    response = handle_message(
        {
            "type": "embed",
            "payload": {
                "batch_id": 7,
                "thumb_px": 400,
                "units": [
                    {"unit_idx": 1, "path": "/tmp/a.jpg"},
                    {"unit_idx": 3, "path": "/tmp/b.mov", "frame_ts": 4.0},
                ],
            },
        }
    )

    assert response["type"] == "embed_response"
    assert response["payload"]["batch_id"] == 7
    assert [item["unit_idx"] for item in response["payload"]["items"]] == [1, 3]
