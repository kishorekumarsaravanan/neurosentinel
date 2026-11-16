"""EDF reader: round-trip, scaling, channel selection."""
import numpy as np
import pytest

from neurosentinel.config import CHANNELS, FS
from neurosentinel.edf import read_edf, select_channels
from conftest import make_edf_bytes


def test_round_trip_recovers_signal_within_resolution(rng):
    x = rng.normal(0, 50, (3, 4 * FS))
    rec = read_edf(make_edf_bytes(x, ["FP1-F7", "F7-T7", "T7-P7"]))
    assert rec.labels == ["FP1-F7", "F7-T7", "T7-P7"]
    assert rec.fs == FS
    assert rec.data.shape == (3, 4 * FS)
    assert rec.duration == pytest.approx(4.0)
    # half a quantisation step (0.05 uV); the gain is 6553.5 / 65535, i.e. a hair above 0.1 uV per count
    assert np.max(np.abs(rec.data - x)) <= 0.051


def test_reads_from_path(tmp_path, rng):
    x = rng.normal(0, 30, (2, 2 * FS))
    p = tmp_path / "rec.edf"
    p.write_bytes(make_edf_bytes(x, ["FZ-CZ", "CZ-PZ"]))
    rec = read_edf(p)
    assert rec.data.shape == (2, 2 * FS)
    assert rec.data.dtype == np.float32


def test_select_channels_orders_by_montage_and_handles_duplicate_labels(rng):
    # CHB-MIT files label the repeated T8-P8 channel as "T8-P8-0" / "T8-P8-1"
    labels = list(reversed(CHANNELS[:14])) + ["T8-P8-0", "P8-O2", "FZ-CZ", "CZ-PZ", "T8-P8-1"]
    x = np.stack([np.full(FS, i, float) for i in range(len(labels))])
    rec = read_edf(make_edf_bytes(x, labels))
    sel = select_channels(rec, CHANNELS)
    assert sel.shape == (len(CHANNELS), FS)
    for row, ch in enumerate(CHANNELS):
        want = next(i for i, lab in enumerate(labels) if lab.split("-0")[0] == ch)  # first match wins
        assert sel[row, 0] == pytest.approx(want, abs=0.05)


def test_select_channels_raises_for_missing_channel(rng):
    rec = read_edf(make_edf_bytes(rng.normal(0, 1, (1, FS)), ["FP1-F7"]))
    with pytest.raises(ValueError, match="F7-T7"):
        select_channels(rec, ["FP1-F7", "F7-T7"])
