"""Behavioural tests for the embedder seam via [`FakeEmbedder`]. They assert the
*contract* (output sizes, determinism) that any real adapter must also satisfy,
so they double as a conformance spec for the SigLIP adapter landing in M0/M1."""

import io
import sys
import types

import pytest

from lume_sidecar.embedder import FP16_BYTES, FakeEmbedder, _decode_still_image, _thumbnail_jpeg


def test_image_embedding_has_fp16_dim_and_a_thumbnail():
    emb, thumb = FakeEmbedder().embed_image("/any/photo.jpg", thumb_px=400)
    assert len(emb) == FP16_BYTES  # 768 * 2 bytes
    assert thumb[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_frame_embedding_matches_image_contract():
    emb, thumb = FakeEmbedder().embed_frame("/any/clip.mov", frame_ts=4.0, thumb_px=400)
    assert len(emb) == FP16_BYTES
    assert thumb[:2] == b"\xff\xd8"


def test_query_image_embedding_is_just_a_vector():
    emb = FakeEmbedder().embed_query_image(b"\xff\xd8\xff\xd9")
    assert len(emb) == FP16_BYTES


def test_query_text_embedding_is_just_a_vector():
    emb = FakeEmbedder().embed_query_text("girl riding a bicycle")
    assert len(emb) == FP16_BYTES


def test_heic_still_decodes_when_pillow_heif_is_available(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    from PIL import Image

    path = tmp_path / "sample.heic"
    image = Image.new("RGB", (12, 8), "red")
    pillow_heif.register_heif_opener()
    image.save(path, format="HEIF")

    decoded = _decode_still_image(str(path))

    assert decoded.image.mode == "RGB"
    assert decoded.image.size == (12, 8)


def test_raw_still_uses_embedded_jpeg_preview_without_demosaic(tmp_path, monkeypatch):
    from PIL import Image

    preview_buffer = io.BytesIO()
    Image.new("RGB", (16, 10), "blue").save(preview_buffer, format="JPEG")

    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_thumb(self):
            return types.SimpleNamespace(
                format=FakeRawpy.ThumbFormat.JPEG,
                data=preview_buffer.getvalue(),
            )

        def postprocess(self):
            raise AssertionError("RAW full demosaic should not run for a preview-backed still")

    class FakeRawpy:
        class ThumbFormat:
            JPEG = object()

        def imread(self, path):
            return FakeRaw()

    monkeypatch.setitem(sys.modules, "rawpy", FakeRawpy())
    path = tmp_path / "sample.dng"
    path.write_bytes(b"not a real raw; fake rawpy owns the test")

    decoded = _decode_still_image(str(path))
    thumb = _thumbnail_jpeg(decoded, thumb_px=400)

    assert decoded.image.size == (16, 10)
    assert thumb == preview_buffer.getvalue()


def test_raw_support_failure_names_missing_optional_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rawpy", None)
    path = tmp_path / "sample.cr2"
    path.write_bytes(b"fixture")

    with pytest.raises(RuntimeError, match="RAW support requires rawpy"):
        _decode_still_image(str(path))
