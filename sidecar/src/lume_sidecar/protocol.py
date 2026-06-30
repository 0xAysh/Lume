"""The Python side of the Rust <-> Python wire contract (DESIGN §9, §19).

This mirrors `crates/ipc/src/protocol.rs`. The same rule applies on both sides:
**nothing about model / device / framework crosses this seam** — only paths out,
and compressed vectors + JPEG thumbnails back. Raw pixels never cross (§6).

`to_dict` / `from_dict` use serde's default *externally-tagged* representation
for the `UnitResult` union (``{"Ok": {...}}`` / ``{"Failed": {...}}``) so the two
sides stay wire-compatible by construction. The concrete framing (length-prefixed
JSON header + binary payload vs. msgpack) is finalized when the transport lands
(BUILD.md M1); until then the round-trip tests pin the shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestUnit:
    """One Unit to embed. ``frame_ts`` None = whole image, else a video frame."""

    unit_idx: int
    path: str
    frame_ts: float | None = None

    def to_dict(self) -> dict:
        d: dict = {"unit_idx": self.unit_idx, "path": self.path}
        if self.frame_ts is not None:
            d["frame_ts"] = self.frame_ts
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RequestUnit:
        return cls(unit_idx=d["unit_idx"], path=d["path"], frame_ts=d.get("frame_ts"))


@dataclass
class EmbedRequest:
    batch_id: int
    units: list[RequestUnit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"batch_id": self.batch_id, "units": [u.to_dict() for u in self.units]}

    @classmethod
    def from_dict(cls, d: dict) -> EmbedRequest:
        return cls(
            batch_id=d["batch_id"],
            units=[RequestUnit.from_dict(u) for u in d["units"]],
        )


@dataclass
class UnitOk:
    """A successful embed: the fp16 vector bytes and the grid thumbnail JPEG."""

    emb_fp16: bytes
    thumb_jpeg: bytes


@dataclass
class UnitFailed:
    """An in-band per-unit failure (corrupt/unsupported file) — DESIGN §17."""

    reason: str


UnitResult = UnitOk | UnitFailed


def _unit_result_to_dict(r: UnitResult) -> dict:
    if isinstance(r, UnitOk):
        return {"Ok": {"emb_fp16": list(r.emb_fp16), "thumb_jpeg": list(r.thumb_jpeg)}}
    return {"Failed": {"reason": r.reason}}


def _unit_result_from_dict(d: dict) -> UnitResult:
    if "Ok" in d:
        ok = d["Ok"]
        return UnitOk(emb_fp16=bytes(ok["emb_fp16"]), thumb_jpeg=bytes(ok["thumb_jpeg"]))
    return UnitFailed(reason=d["Failed"]["reason"])


@dataclass
class BatchItem:
    unit_idx: int
    result: UnitResult

    def to_dict(self) -> dict:
        return {"unit_idx": self.unit_idx, "result": _unit_result_to_dict(self.result)}

    @classmethod
    def from_dict(cls, d: dict) -> BatchItem:
        return cls(unit_idx=d["unit_idx"], result=_unit_result_from_dict(d["result"]))


@dataclass
class EmbedResponse:
    batch_id: int
    items: list[BatchItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"batch_id": self.batch_id, "items": [i.to_dict() for i in self.items]}

    @classmethod
    def from_dict(cls, d: dict) -> EmbedResponse:
        return cls(
            batch_id=d["batch_id"],
            items=[BatchItem.from_dict(i) for i in d["items"]],
        )
