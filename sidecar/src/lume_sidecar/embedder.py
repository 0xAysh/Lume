"""The embedder seam: ``path -> (embedding, thumbnail)``.

This is the Python-internal interface the socket server calls. The real adapter
(SigLIP 2 Base on MPS) is deferred to the M0 spike; tests run against
[`FakeEmbedder`]. Keeping this an ABC is what lets the whole ONNX / Core ML
future be a Python-only swap behind the socket black box (DESIGN §19, §21).

**Tower-separation requirement (DESIGN §11, build into M1):** the real adapter
must load the SigLIP text and vision towers *independently* (``SiglipTextModel``
+ ``SiglipVisionModel``, not only the monolithic ``SiglipModel``) so M5 can
"unload the vision tower on idle, keep the text encoder resident" without
surgery on a monolithic model object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# 768-dim fp16 (DESIGN §8). Vector dtype is config-swappable on the Rust side,
# but the sidecar always emits the model's native dimensionality.
EMBED_DIM = 768
FP16_BYTES = EMBED_DIM * 2


class Embedder(ABC):
    """Decode + preprocess + embed. The sidecar's black-box core."""

    @abstractmethod
    def embed_image(self, path: str) -> tuple[bytes, bytes]:
        """Embed a whole image. Returns ``(emb_fp16, thumb_jpeg)``.

        Raises on a decode/embed failure; the caller turns that into an in-band
        ``UnitFailed`` (DESIGN §17), never a transport error.
        """

    @abstractmethod
    def embed_frame(self, path: str, frame_ts: float) -> tuple[bytes, bytes]:
        """Embed one video frame at ``frame_ts`` seconds."""

    @abstractmethod
    def embed_query_image(self, image_bytes: bytes) -> bytes:
        """Synchronous drag-in path: embed in-memory image bytes -> ``emb_fp16``."""

    @abstractmethod
    def embed_query_text(self, text: str) -> bytes:
        """Synchronous search path: embed query text -> ``emb_fp16``.

        The real adapter serves this from the resident text tower (DESIGN §11),
        so search-after-idle stays instant while the vision tower can unload.
        """


class FakeEmbedder(Embedder):
    """Deterministic stand-in for tests — no torch, no GPU, no decode.

    Produces a fixed-length zero vector and a tiny stub JPEG so the indexing and
    socket paths can be exercised without the multi-GB model (DESIGN §19: the
    seam exists so the slow collaborator can be faked).
    """

    _STUB_JPEG = bytes([0xFF, 0xD8, 0xFF, 0xD9])  # SOI + EOI: smallest JPEG marker pair

    def embed_image(self, path: str) -> tuple[bytes, bytes]:
        return (bytes(FP16_BYTES), self._STUB_JPEG)

    def embed_frame(self, path: str, frame_ts: float) -> tuple[bytes, bytes]:
        return (bytes(FP16_BYTES), self._STUB_JPEG)

    def embed_query_image(self, image_bytes: bytes) -> bytes:
        return bytes(FP16_BYTES)

    def embed_query_text(self, text: str) -> bytes:
        return bytes(FP16_BYTES)
