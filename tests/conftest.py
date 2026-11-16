"""Shared fixtures: synthetic EEG, a minimal EDF writer and paths to optional real data."""
from __future__ import annotations
import numpy as np
import pytest

from neurosentinel.config import CHANNELS, FS, DATA_RAW, ARTIFACTS


def make_edf_bytes(signals: np.ndarray, labels: list[str], fs: int = FS, record_sec: int = 1) -> bytes:
    """Write a minimal, spec-compliant EDF (16-bit) file in memory.

    Physical range +/-3276.8 uV over the full int16 range gives a 0.1 uV resolution.
    """
    ns, n = signals.shape
    spr = fs * record_sec
    n_rec = n // spr
    f = lambda v, w: str(v).ljust(w)[:w].encode("latin-1")
    head = (f("0", 8) + f("test patient", 80) + f("test recording", 80) + f("01.01.01", 8) + f("00.00.00", 8)
            + f(256 + 256 * ns, 8) + f("", 44) + f(n_rec, 8) + f(record_sec, 8) + f(ns, 4))
    fields = [(16, labels), (80, [""] * ns), (8, ["uV"] * ns), (8, ["-3276.8"] * ns), (8, ["3276.7"] * ns),
              (8, ["-32768"] * ns), (8, ["32767"] * ns), (80, [""] * ns), (8, [spr] * ns), (32, [""] * ns)]
    for width, vals in fields:
        head += b"".join(f(v, width) for v in vals)
    dig = np.clip(np.round(signals[:, :n_rec * spr] * 10), -32768, 32767).astype("<i2")
    body = dig.reshape(ns, n_rec, spr).transpose(1, 0, 2).tobytes()
    return head + body


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def background_eeg(rng):
    """10 s of 18-channel 'background': pink-ish noise around 20 uV."""
    x = rng.normal(0, 20, (len(CHANNELS), 10 * FS))
    return np.cumsum(x, axis=1) * 0.05 + rng.normal(0, 10, x.shape)


@pytest.fixture
def seizure_like_eeg(rng):
    """10 s of 18-channel 'ictal' activity: large 3 Hz rhythmic discharges plus noise."""
    t = np.arange(10 * FS) / FS
    return 150 * np.sin(2 * np.pi * 3 * t)[None, :] + rng.normal(0, 10, (len(CHANNELS), t.size))


def has_raw_data() -> bool:
    return any(DATA_RAW.glob("chb*/*.edf"))


def has_artifacts() -> bool:
    return (ARTIFACTS / "results.json").exists()


needs_data = pytest.mark.skipif(not has_raw_data(), reason="CHB-MIT EDF files not downloaded (bash scripts/download_data.sh)")
needs_artifacts = pytest.mark.skipif(not has_artifacts(), reason="artifacts/ missing (run python train.py)")
