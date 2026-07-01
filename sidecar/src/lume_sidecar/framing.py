"""Length-prefixed JSON frames for the Sidecar socket."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

MAX_FRAME_BYTES = 32 * 1024 * 1024


def write_frame(sock: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {len(payload)} bytes")
    sock.sendall(struct.pack(">I", len(payload)))
    sock.sendall(payload)


def read_frame(sock: socket.socket) -> dict[str, Any]:
    length_bytes = _recv_exact(sock, 4)
    length = struct.unpack(">I", length_bytes)[0]
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {length} bytes")
    return json.loads(_recv_exact(sock, length).decode("utf-8"))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(n - len(chunks))
        if not chunk:
            raise EOFError("socket closed mid-frame")
        chunks.extend(chunk)
    return bytes(chunks)
