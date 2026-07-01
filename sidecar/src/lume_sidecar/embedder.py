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

import io
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

# 768-dim fp16 (DESIGN §8). Vector dtype is config-swappable on the Rust side,
# but the sidecar always emits the model's native dimensionality.
EMBED_DIM = 768
FP16_BYTES = EMBED_DIM * 2


class Embedder(ABC):
    """Decode + preprocess + embed. The sidecar's black-box core."""

    @abstractmethod
    def embed_image(self, path: str, thumb_px: int) -> tuple[bytes, bytes]:
        """Embed a whole image. Returns ``(emb_fp16, thumb_jpeg)``.

        ``thumb_px`` sizes the returned grid thumbnail on its longest edge
        (`Config.thumbnails.grid_px`, DESIGN §8).

        Raises on a decode/embed failure; the caller turns that into an in-band
        ``UnitFailed`` (DESIGN §17), never a transport error.
        """

    @abstractmethod
    def embed_frame(self, path: str, frame_ts: float, thumb_px: int) -> tuple[bytes, bytes]:
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

    def embed_image(self, path: str, thumb_px: int) -> tuple[bytes, bytes]:
        return (bytes(FP16_BYTES), self._STUB_JPEG)

    def embed_frame(self, path: str, frame_ts: float, thumb_px: int) -> tuple[bytes, bytes]:
        return (bytes(FP16_BYTES), self._STUB_JPEG)

    def embed_query_image(self, image_bytes: bytes) -> bytes:
        return bytes(FP16_BYTES)

    def embed_query_text(self, text: str) -> bytes:
        return bytes(FP16_BYTES)


class SiglipEmbedder(Embedder):
    """Real SigLIP 2 adapter: independently-loaded vision/text towers on MPS
    (falling back to CPU), matching the M0 probe's proven loading pattern
    (`scripts/m0_siglip_mps_probe.py`).

    Heavy ML deps (torch, transformers, Pillow — the `m0` uv dependency
    group) are imported lazily in ``__init__``, not at module scope. Merely
    importing this module — which `server.py` always does, including under
    `make test-py` — never pulls them in; only constructing a
    ``SiglipEmbedder`` does. This is what keeps normal tests fast and
    dependency-light (mirrors why the MPS probe lives outside `make test-py`).
    """

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            SiglipTextModel,
            SiglipVisionModel,
        )

        self._torch = torch
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self._device == "cpu":
            logger.info("SiglipEmbedder: MPS unavailable, falling back to CPU")

        self._image_processor = AutoImageProcessor.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._vision = SiglipVisionModel.from_pretrained(model_name).to(self._device).eval()
        self._text = SiglipTextModel.from_pretrained(model_name).to(self._device).eval()

    def embed_image(self, path: str, thumb_px: int) -> tuple[bytes, bytes]:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        return (self._embed_pixels(image), _thumbnail_jpeg(image, thumb_px))

    def embed_frame(self, path: str, frame_ts: float, thumb_px: int) -> tuple[bytes, bytes]:
        raise NotImplementedError("video frame embedding lands in M3")

    def embed_query_image(self, image_bytes: bytes) -> bytes:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._embed_pixels(image)

    def embed_query_text(self, text: str) -> bytes:
        torch = self._torch
        inputs = self._tokenizer([text], padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            pooled = self._text(**inputs).pooler_output
        return _normalized_fp16_bytes(torch, pooled)

    def _embed_pixels(self, image: PILImage) -> bytes:
        torch = self._torch
        inputs = self._image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            pooled = self._vision(**inputs).pooler_output
        return _normalized_fp16_bytes(torch, pooled)


def _normalized_fp16_bytes(torch, pooled) -> bytes:
    """L2-normalize (standard SigLIP practice — makes cosine similarity a
    simple L2-distance computation store-side, ADR-0003) and pack as
    little-endian fp16 bytes, matching `lume_ipc`'s `f16::from_le_bytes` read."""
    normalized = torch.nn.functional.normalize(pooled, dim=-1)
    array = normalized.to("cpu").to(torch.float16).detach().numpy()
    return array.astype("<f2").tobytes()


def _thumbnail_jpeg(image: PILImage, thumb_px: int) -> bytes:
    from PIL import Image

    resized = image.copy()
    resized.thumbnail((thumb_px, thumb_px), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()
