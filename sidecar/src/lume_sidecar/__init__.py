"""Lume sidecar — the inference black box (DESIGN §6, §9, §19).

A separate process that owns *all* decode, preprocessing, and SigLIP 2 embedding,
reached from Rust over a Unix socket as ``path -> (embedding, thumbnail)``.
Nothing about model / device / framework crosses the socket.
"""

from lume_sidecar.embedder import EMBED_DIM, Embedder, FakeEmbedder

__all__ = ["EMBED_DIM", "Embedder", "FakeEmbedder"]
