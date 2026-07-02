"""Unix-socket Sidecar server: multiplexes bulk/embed-one/embed-text requests
onto the real `SiglipEmbedder` by default, with a `--fake` override for tests
(DESIGN §9, §19; BUILD.md M1 Slice 2)."""

from __future__ import annotations

import argparse
import os
import socket
import threading
from pathlib import Path
from typing import Any

from lume_sidecar.batching import BatchPipeline, SidecarScheduler
from lume_sidecar.embedder import Embedder, FakeEmbedder, SiglipEmbedder
from lume_sidecar.framing import read_frame, write_frame


def handle_message(
    message: dict[str, Any],
    embedder: Embedder | None = None,
    *,
    decode_workers: int | None = None,
) -> dict[str, Any]:
    embedder = embedder or FakeEmbedder()
    return BatchPipeline(embedder, decode_workers).handle_message(message)


def serve(
    socket_path: Path,
    embedder: Embedder | None = None,
    *,
    decode_workers: int | None = None,
    max_requests: int | None = None,
) -> None:
    if socket_path.exists():
        socket_path.unlink()
    embedder = embedder or FakeEmbedder()
    scheduler = SidecarScheduler(embedder, decode_workers)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    handlers: list[threading.Thread] = []
    try:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen()
        accepted = 0
        while max_requests is None or accepted < max_requests:
            conn, _ = server.accept()
            accepted += 1
            handler = threading.Thread(
                target=_handle_connection,
                args=(conn, scheduler),
                name=f"lume-sidecar-client-{accepted}",
            )
            handler.start()
            if max_requests is not None:
                handlers.append(handler)

        for handler in handlers:
            handler.join()
    finally:
        scheduler.shutdown()
        server.close()
        if socket_path.exists():
            socket_path.unlink()


def _handle_connection(conn: socket.socket, scheduler: SidecarScheduler) -> None:
    with conn:
        try:
            future = scheduler.submit(read_frame(conn))
            write_frame(conn, future.result())
        except Exception as exc:  # noqa: BLE001 - transport error response for M0 probe.
            write_frame(conn, {"type": "error", "payload": {"message": str(exc)}})


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
