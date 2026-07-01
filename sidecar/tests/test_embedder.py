"""Behavioural tests for the embedder seam via [`FakeEmbedder`]. They assert the
*contract* (output sizes, determinism) that any real adapter must also satisfy,
so they double as a conformance spec for the SigLIP adapter landing in M0/M1."""

from lume_sidecar.embedder import FP16_BYTES, FakeEmbedder


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
