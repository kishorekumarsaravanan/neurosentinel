"""Filtering and segmentation."""
import numpy as np
import pytest

from neurosentinel.config import FS, WIN
from neurosentinel.preprocess import bandpass, notch, preprocess, window_starts, make_windows, seizure_overlap

t = np.arange(20 * FS) / FS


def amplitude(x):
    return np.sqrt(2) * np.std(x[FS:-FS])          # ignore filter edge effects


def test_bandpass_keeps_in_band_and_removes_out_of_band():
    in_band = np.sin(2 * np.pi * 10 * t)
    drift = np.sin(2 * np.pi * 0.05 * t)
    high = np.sin(2 * np.pi * 80 * t)
    assert amplitude(bandpass(in_band, 0.5, 40)) == pytest.approx(1.0, rel=0.02)
    assert amplitude(bandpass(drift, 0.5, 40)) < 0.05
    assert amplitude(bandpass(high, 0.5, 40)) < 0.05    # one octave above cut-off: > 26 dB down


def test_notch_removes_mains_hum_only():
    hum = np.sin(2 * np.pi * 60 * t)
    alpha = np.sin(2 * np.pi * 10 * t)
    assert amplitude(notch(hum)) < 0.05
    assert amplitude(notch(alpha)) == pytest.approx(1.0, rel=0.01)


def test_preprocess_is_zero_phase_and_float32():
    x = np.sin(2 * np.pi * 8 * t)[None, :].repeat(3, 0)
    y = preprocess(x)
    assert y.dtype == np.float32 and y.shape == x.shape
    # zero-phase filtering: no lag between input and output peaks
    lag = np.argmax(np.correlate(y[0, 2000:3000], x[0, 2000:3000], "full")) - 999
    assert abs(lag) <= 1


def test_window_starts_and_make_windows():
    n = 10 * FS
    st = window_starts(n, FS)
    assert st[0] == 0 and st[-1] == n - WIN and np.all(np.diff(st) == FS)
    x = np.arange(2 * n, dtype=float).reshape(2, n)
    W = make_windows(x, st[:3])
    assert W.shape == (3, 2, WIN)
    assert W[1, 0, 0] == FS and W[1, 1, 0] == n + FS


def test_seizure_overlap_fraction():
    mask = np.zeros(20 * FS, bool)
    mask[8 * FS:12 * FS] = True                       # seizure from 8 s to 12 s
    st = np.array([0, 6 * FS, 8 * FS, 10 * FS])
    np.testing.assert_allclose(seizure_overlap(st, mask), [0.0, 0.5, 1.0, 0.5])
