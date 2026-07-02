"""Sidecar-internal batching and request scheduling.

The Rust seam stays small: it sends one ``embed`` request per configured batch
and separate interactive ``embed_text`` / ``embed_one`` requests. This module
deepens the Python black box behind that seam: CPU still-image decode can run in
parallel, all model inference stays serialized on one lane, and queued
interactive work is preferred at bulk-batch boundaries.
"""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from lume_sidecar.embedder import (
    DecodedStill,
    Embedder,
    _decode_still_image,
    _thumbnail_jpeg,
)
from lume_sidecar.protocol import (
    BatchItem,
    EmbedOneRequest,
    EmbedOneResponse,
    EmbedRequest,
    EmbedResponse,
    EmbedTextRequest,
    RequestUnit,
    UnitFailed,
    UnitOk,
)


def default_decode_workers() -> int:
    """Small bounded pool: enough to overlap CPU decode without ballooning RAM."""

    return max(1, min(4, os.cpu_count() or 1))


@dataclass(frozen=True)
class _DecodedUnit:
    unit_idx: int
    image: Any
    thumb_jpeg: bytes


@dataclass(frozen=True)
class _QueuedJob:
    message: dict[str, Any]
    future: Future[dict[str, Any]]


class BatchPipeline:
    """Executes one Sidecar request on the serialized inference lane."""

    def __init__(self, embedder: Embedder, decode_workers: int | None = None) -> None:
        self._embedder = embedder
        self._decode_workers = decode_workers or default_decode_workers()

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        message_type = message["type"]
        payload = message["payload"]

        if message_type == "embed":
            req = EmbedRequest.from_dict(payload)
            return {
                "type": "embed_response",
                "payload": self.embed_batch(req).to_dict(),
            }

        if message_type == "embed_one":
            req = EmbedOneRequest.from_dict(payload)
            return {
                "type": "embed_one_response",
                "payload": EmbedOneResponse(
                    emb_fp16=self._embedder.embed_query_image(req.image_bytes)
                ).to_dict(),
            }

        if message_type == "embed_text":
            req = EmbedTextRequest.from_dict(payload)
            return {
                "type": "embed_one_response",
                "payload": EmbedOneResponse(
                    emb_fp16=self._embedder.embed_query_text(req.text)
                ).to_dict(),
            }

        return {"type": "error", "payload": {"message": f"unknown message type: {message_type}"}}

    def embed_batch(self, req: EmbedRequest) -> EmbedResponse:
        if _supports_decoded_still_batch(self._embedder):
            return self._embed_decoded_still_batch(req)
        return self._embed_fallback_batch(req)

    def _embed_fallback_batch(self, req: EmbedRequest) -> EmbedResponse:
        items: list[BatchItem] = []
        for unit in req.units:
            try:
                if unit.frame_ts is None:
                    emb, thumb = self._embedder.embed_image(unit.path, req.thumb_px)
                else:
                    emb, thumb = self._embedder.embed_frame(unit.path, unit.frame_ts, req.thumb_px)
                result = UnitOk(emb_fp16=emb, thumb_jpeg=thumb)
            except Exception as exc:  # noqa: BLE001 - in-band per-Unit failure by design.
                result = UnitFailed(reason=str(exc))
            items.append(BatchItem(unit_idx=unit.unit_idx, result=result))
        return EmbedResponse(batch_id=req.batch_id, items=items)

    def _embed_decoded_still_batch(self, req: EmbedRequest) -> EmbedResponse:
        decoded: list[_DecodedUnit] = []
        items: list[BatchItem] = []

        still_units = [unit for unit in req.units if unit.frame_ts is None]
        frame_units = [unit for unit in req.units if unit.frame_ts is not None]

        if still_units:
            with ThreadPoolExecutor(max_workers=self._decode_workers) as pool:
                futures = {
                    pool.submit(_decode_unit, unit, req.thumb_px): unit for unit in still_units
                }
                for future in as_completed(futures):
                    unit = futures[future]
                    try:
                        decoded.append(future.result())
                    except Exception as exc:  # noqa: BLE001 - in-band per-Unit failure by design.
                        items.append(
                            BatchItem(unit_idx=unit.unit_idx, result=UnitFailed(reason=str(exc)))
                        )

        if decoded:
            images = [unit.image for unit in decoded]
            embeddings = self._embedder.embed_decoded_stills(images)  # type: ignore[attr-defined]
            for unit, emb in zip(decoded, embeddings, strict=True):
                items.append(
                    BatchItem(
                        unit_idx=unit.unit_idx,
                        result=UnitOk(emb_fp16=emb, thumb_jpeg=unit.thumb_jpeg),
                    )
                )

        for unit in frame_units:
            try:
                emb, thumb = self._embedder.embed_frame(unit.path, unit.frame_ts, req.thumb_px)
                result = UnitOk(emb_fp16=emb, thumb_jpeg=thumb)
            except Exception as exc:  # noqa: BLE001 - in-band per-Unit failure by design.
                result = UnitFailed(reason=str(exc))
            items.append(BatchItem(unit_idx=unit.unit_idx, result=result))

        return EmbedResponse(batch_id=req.batch_id, items=items)


class SidecarScheduler:
    """One serialized inference lane with interactive priority between batches."""

    def __init__(self, embedder: Embedder, decode_workers: int | None = None) -> None:
        self._pipeline = BatchPipeline(embedder, decode_workers)
        self._bulk: queue.Queue[_QueuedJob] = queue.Queue()
        self._interactive: queue.Queue[_QueuedJob] = queue.Queue()
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lume-sidecar-scheduler")
        self._thread.start()

    def submit(self, message: dict[str, Any]) -> Future[dict[str, Any]]:
        future: Future[dict[str, Any]] = Future()
        job = _QueuedJob(message=message, future=future)
        if message.get("type") == "embed":
            self._bulk.put(job)
        else:
            self._interactive.put(job)
        return future

    def shutdown(self) -> None:
        self._stopping.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stopping.is_set() or not self._interactive.empty() or not self._bulk.empty():
            job = self._next_job()
            if job is None:
                continue
            self._complete(job)

    def _next_job(self) -> _QueuedJob | None:
        try:
            return self._interactive.get_nowait()
        except queue.Empty:
            pass

        try:
            return self._bulk.get(timeout=0.01)
        except queue.Empty:
            return None

    def _complete(self, job: _QueuedJob) -> None:
        try:
            job.future.set_result(self._pipeline.handle_message(job.message))
        except Exception as exc:  # noqa: BLE001 - transport-level error for malformed requests.
            job.future.set_result({"type": "error", "payload": {"message": str(exc)}})


def _supports_decoded_still_batch(embedder: Embedder) -> bool:
    return callable(getattr(embedder, "embed_decoded_stills", None))


def _decode_unit(unit: RequestUnit, thumb_px: int) -> _DecodedUnit:
    decoded = _decode_still_image(unit.path)
    return _DecodedUnit(
        unit_idx=unit.unit_idx,
        image=decoded.image,
        thumb_jpeg=_thumbnail_jpeg(decoded, thumb_px),
    )
