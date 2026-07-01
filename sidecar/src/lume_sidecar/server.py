"""Unix-socket Sidecar server: multiplexes bulk/embed-one/embed-text requests
onto the real `SiglipEmbedder` by default, with a `--fake` override for tests
(DESIGN §9, §19; BUILD.md M1 Slice 2)."""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from typing import Any

from lume_sidecar.embedder import Embedder, FakeEmbedder, SiglipEmbedder
from lume_sidecar.framing import read_frame, write_frame
from lume_sidecar.protocol import (
    BatchItem,
    EmbedOneRequest,
    EmbedOneResponse,
    EmbedRequest,
    EmbedResponse,
    EmbedTextRequest,
    UnitFailed,
    UnitOk,
)


def handle_message(message: dict[str, Any], embedder: Embedder | None = None) -> dict[str, Any]:
    embedder = embedder or FakeEmbedder()
    message_type = message["type"]
    payload = message["payload"]

    if message_type == "embed":
        req = EmbedRequest.from_dict(payload)
        items: list[BatchItem] = []
        for unit in req.units:
            try:
                if unit.frame_ts is None:
                    emb, thumb = embedder.embed_image(unit.path, req.thumb_px)
                else:
                    emb, thumb = embedder.embed_frame(unit.path, unit.frame_ts, req.thumb_px)
                result = UnitOk(emb_fp16=emb, thumb_jpeg=thumb)
            except Exception as exc:  # noqa: BLE001 - in-band per-Unit failure by design.
                result = UnitFailed(reason=str(exc))
            items.append(BatchItem(unit_idx=unit.unit_idx, result=result))
        return {
            "type": "embed_response",
            "payload": EmbedResponse(batch_id=req.batch_id, items=items).to_dict(),
        }

    if message_type == "embed_one":
        req = EmbedOneRequest.from_dict(payload)
        return {
            "type": "embed_one_response",
            "payload": EmbedOneResponse(
                emb_fp16=embedder.embed_query_image(req.image_bytes)
            ).to_dict(),
        }

    if message_type == "embed_text":
        req = EmbedTextRequest.from_dict(payload)
        return {
            "type": "embed_one_response",
            "payload": EmbedOneResponse(emb_fp16=embedder.embed_query_text(req.text)).to_dict(),
        }

    return {"type": "error", "payload": {"message": f"unknown message type: {message_type}"}}


def serve(socket_path: Path, embedder: Embedder | None = None) -> None:
    if socket_path.exists():
        socket_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen()
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    write_frame(conn, handle_message(read_frame(conn), embedder))
                except Exception as exc:  # noqa: BLE001 - transport error response for M0 probe.
                    write_frame(conn, {"type": "error", "payload": {"message": str(exc)}})
    finally:
        server.close()
        if socket_path.exists():
            socket_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Lume Sidecar socket server.")
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use FakeEmbedder instead of loading real SigLIP weights "
        "(fast startup, no GPU/model download — used by tests).",
    )
    args = parser.parse_args()

    if args.fake or os.environ.get("LUME_SIDECAR_FAKE_EMBEDDER"):
        embedder: Embedder = FakeEmbedder()
    else:
        embedder = SiglipEmbedder(args.model)

    serve(args.socket, embedder)


if __name__ == "__main__":
    main()
