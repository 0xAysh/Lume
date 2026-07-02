import socket
import os
import threading
import time
from pathlib import Path

from PIL import Image

from lume_sidecar.embedder import FP16_BYTES, Embedder
from lume_sidecar.framing import read_frame, write_frame
from lume_sidecar.protocol import (
    BatchItem,
    EmbedOneResponse,
    EmbedRequest,
    EmbedResponse,
    RequestUnit,
    UnitFailed,
    UnitOk,
)
from lume_sidecar.server import handle_message, serve


class BatchCapableProbeEmbedder(Embedder):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed_image(self, path: str, thumb_px: int) -> tuple[bytes, bytes]:
        raise AssertionError("bulk stills should use the decoded batch path")

    def embed_frame(self, path: str, frame_ts: float, thumb_px: int) -> tuple[bytes, bytes]:
        raise AssertionError("not used by this test")

    def embed_query_image(self, image_bytes: bytes) -> bytes:
        return b"\x04" * FP16_BYTES

    def embed_query_text(self, text: str) -> bytes:
        return b"\x05" * FP16_BYTES

    def embed_decoded_stills(self, images):
        self.batch_sizes.append(len(images))
        return [bytes([idx + 1]) * FP16_BYTES for idx, _image in enumerate(images)]


def test_bulk_embed_decodes_in_workers_then_runs_one_ordered_main_thread_batch(tmp_path):
    slow = tmp_path / "slow.jpg"
    fast = tmp_path / "fast.jpg"
    broken = tmp_path / "broken.jpg"
    Image.new("RGB", (8, 8), "red").save(slow)
    Image.new("RGB", (8, 8), "blue").save(fast)
    broken.write_bytes(b"not an image")

    embedder = BatchCapableProbeEmbedder()
    message = {
        "type": "embed",
        "payload": EmbedRequest(
            batch_id=7,
            thumb_px=16,
            units=[
                RequestUnit(unit_idx=10, path=str(slow)),
                RequestUnit(unit_idx=3, path=str(broken)),
                RequestUnit(unit_idx=5, path=str(fast)),
            ],
        ).to_dict(),
    }

    response = handle_message(message, embedder, decode_workers=2)

    payload = EmbedResponse.from_dict(response["payload"])
    assert payload.batch_id == 7
    assert embedder.batch_sizes == [2]
    assert [item.unit_idx for item in payload.items] == [3, 10, 5]
    by_idx = {item.unit_idx: item.result for item in payload.items}
    assert isinstance(by_idx[3], UnitFailed)
    assert isinstance(by_idx[10], UnitOk)
    assert isinstance(by_idx[5], UnitOk)
    assert by_idx[10].thumb_jpeg.startswith(b"\xff\xd8")


class BoundaryProbeEmbedder(Embedder):
    def __init__(self) -> None:
        self.events: list[str] = []

    def embed_image(self, path: str, thumb_px: int) -> tuple[bytes, bytes]:
        self.events.append(f"bulk:{Path(path).name}")
        time.sleep(0.05)
        return b"\x01" * FP16_BYTES, b"\xff\xd8\xff\xd9"

    def embed_frame(self, path: str, frame_ts: float, thumb_px: int) -> tuple[bytes, bytes]:
        raise AssertionError("not used by this test")

    def embed_query_image(self, image_bytes: bytes) -> bytes:
        self.events.append("image-query")
        return b"\x02" * FP16_BYTES

    def embed_query_text(self, text: str) -> bytes:
        self.events.append(f"text:{text}")
        return b"\x03" * FP16_BYTES


def _round_trip(socket_path: Path, message: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        write_frame(client, message)
        return read_frame(client)


def test_server_drains_waiting_interactive_request_between_bulk_batches(tmp_path):
    socket_path = Path(f"/tmp/lume-sidecar-test-{os.getpid()}.sock")
    if socket_path.exists():
        socket_path.unlink()
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"fixture")
    second.write_bytes(b"fixture")
    embedder = BoundaryProbeEmbedder()

    server = threading.Thread(
        target=serve,
        kwargs={"socket_path": socket_path, "embedder": embedder, "max_requests": 4},
    )
    server.start()
    deadline = time.time() + 2
    while not socket_path.exists() and time.time() < deadline:
        time.sleep(0.005)

    first_response: dict | None = None

    def send_first_batch() -> None:
        nonlocal first_response
        first_response = _round_trip(
            socket_path,
            {
                "type": "embed",
                "payload": EmbedRequest(
                    batch_id=1,
                    thumb_px=16,
                    units=[RequestUnit(unit_idx=0, path=str(first))],
                ).to_dict(),
            },
        )

    first_thread = threading.Thread(target=send_first_batch)
    first_thread.start()
    time.sleep(0.01)

    text_response: dict | None = None

    def send_text_query() -> None:
        nonlocal text_response
        text_response = _round_trip(
            socket_path,
            {"type": "embed_text", "payload": {"text": "waiting query"}},
        )

    text_thread = threading.Thread(target=send_text_query)
    text_thread.start()

    image_response: dict | None = None

    def send_image_query() -> None:
        nonlocal image_response
        image_response = _round_trip(
            socket_path,
            {"type": "embed_one", "payload": {"image_bytes": "/9j/"}},
        )

    image_thread = threading.Thread(target=send_image_query)
    image_thread.start()
    first_thread.join(timeout=2)
    assert first_response is not None

    second_response = _round_trip(
        socket_path,
        {
            "type": "embed",
            "payload": EmbedRequest(
                batch_id=2,
                thumb_px=16,
                units=[RequestUnit(unit_idx=0, path=str(second))],
            ).to_dict(),
        },
    )

    text_thread.join(timeout=2)
    image_thread.join(timeout=2)
    server.join(timeout=2)

    assert text_response is not None
    assert image_response is not None
    assert second_response["type"] == "embed_response"
    assert EmbedOneResponse.from_dict(text_response["payload"])
    assert EmbedOneResponse.from_dict(image_response["payload"])
    assert embedder.events[:1] == ["bulk:first.jpg"]
    assert set(embedder.events[1:3]) == {"text:waiting query", "image-query"}
    assert embedder.events[3:] == ["bulk:second.jpg"]
