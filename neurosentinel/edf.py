"""Minimal, dependency-free EDF reader (sufficient for CHB-MIT files)."""
from __future__ import annotations
import io
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class EDFRecording:
    labels: list[str]
    fs: float
    data: np.ndarray          # (n_channels, n_samples) float32 in microvolts
    start_time: str = ""

    @property
    def duration(self) -> float:
        return self.data.shape[1] / self.fs


def _field(buf: bytes, offset: int, size: int, count: int) -> tuple[list[str], int]:
    out = [buf[offset + i * size: offset + (i + 1) * size].decode("latin-1").strip() for i in range(count)]
    return out, offset + size * count


def read_edf(source: str | Path | bytes) -> EDFRecording:
    raw = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    head = raw[:256].decode("latin-1")
    header_bytes = int(head[184:192])
    n_records = int(head[236:244])
    rec_dur = float(head[244:252])
    ns = int(head[252:256])
    start_time = head[176:184].strip()

    off = 256
    labels, off = _field(raw, off, 16, ns)
    _, off = _field(raw, off, 80, ns)           # transducer
    _, off = _field(raw, off, 8, ns)            # physical dimension
    pmin, off = _field(raw, off, 8, ns)
    pmax, off = _field(raw, off, 8, ns)
    dmin, off = _field(raw, off, 8, ns)
    dmax, off = _field(raw, off, 8, ns)
    _, off = _field(raw, off, 80, ns)           # prefiltering
    nsamp, off = _field(raw, off, 8, ns)
    nsamp = np.array([int(v) for v in nsamp])
    if not np.all(nsamp == nsamp[0]):
        raise ValueError("Mixed sampling rates are not supported")
    spr = int(nsamp[0])

    body = np.frombuffer(raw, dtype="<i2", offset=header_bytes)
    rec_len = ns * spr
    if n_records <= 0:
        n_records = body.size // rec_len
    body = body[: n_records * rec_len].reshape(n_records, ns, spr)
    dig = body.transpose(1, 0, 2).reshape(ns, n_records * spr).astype(np.float32)

    pmin = np.array(pmin, float); pmax = np.array(pmax, float)
    dmin = np.array(dmin, float); dmax = np.array(dmax, float)
    gain = ((pmax - pmin) / np.where(dmax - dmin == 0, 1, dmax - dmin)).astype(np.float32)
    offset = (pmin - dmin * gain).astype(np.float32)
    data = dig * gain[:, None] + offset[:, None]
    return EDFRecording(labels=labels, fs=spr / rec_dur, data=data, start_time=start_time)


def select_channels(rec: EDFRecording, wanted: list[str]) -> np.ndarray:
    """Return (len(wanted), n_samples) picking the first channel whose label matches."""
    norm = [lab.upper().split("-0")[0].split("-1")[0] if lab.upper().count("-") > 1 else lab.upper()
            for lab in rec.labels]
    rows = []
    for ch in wanted:
        try:
            rows.append(norm.index(ch.upper()))
        except ValueError as exc:
            raise ValueError(f"Channel {ch} not found in {rec.labels}") from exc
    return rec.data[rows]
