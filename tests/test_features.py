"""Hand-crafted features: shape, naming and physiological sanity."""
import numpy as np
import pytest

from neurosentinel.config import CHANNELS, FS, WIN
from neurosentinel.features import (extract_features, band_powers, FEATURE_NAMES, PER_CHANNEL,
                                    TIME_FEATURES, FREQ_FEATURES, FEATURE_FAMILY)
from neurosentinel.preprocess import make_windows, preprocess

NF = len(PER_CHANNEL)


def feats_of(x):
    """Features of the first 4-s window of a (18, n) signal, reshaped to (18, n_features)."""
    W = make_windows(preprocess(x), np.array([2 * FS]))
    return extract_features(W).reshape(len(CHANNELS), NF)


def test_feature_inventory():
    assert NF == 21 and len(TIME_FEATURES) == 8 and len(FREQ_FEATURES) == 13
    assert len(FEATURE_NAMES) == len(CHANNELS) * NF == 378
    assert len(set(FEATURE_NAMES)) == 378
    assert FEATURE_NAMES[0] == "FP1-F7|log_variance"
    assert set(FEATURE_FAMILY.values()) == {"time", "frequency"}


def test_output_shape_dtype_and_finite(background_eeg):
    W = make_windows(background_eeg, np.arange(0, 6 * FS, FS))
    F = extract_features(W)
    assert F.shape == (len(W), 378) and F.dtype == np.float32
    assert np.isfinite(F).all()


def test_empty_input_and_chunking_are_consistent(background_eeg):
    assert extract_features(np.zeros((0, 18, WIN), np.float32)).shape == (0, 378)
    W = make_windows(background_eeg, np.arange(0, 6 * FS, FS))
    np.testing.assert_allclose(extract_features(W, chunk=2), extract_features(W, chunk=512), rtol=1e-5)


def test_alpha_sine_dominates_alpha_band():
    t = np.arange(8 * FS) / FS
    x = np.tile(50 * np.sin(2 * np.pi * 10 * t), (18, 1))
    F = feats_of(x)
    rel = F[:, PER_CHANNEL.index("rel_power_alpha")]
    peak = F[:, PER_CHANNEL.index("peak_frequency")]
    assert np.all(rel > 0.9)
    np.testing.assert_allclose(peak, 10, atol=0.5)
    bp = band_powers(make_windows(preprocess(x), np.array([2 * FS])))
    assert bp.shape == (1, 18, 5) and np.allclose(bp.sum(-1), 1, atol=0.05)


def test_seizure_like_activity_raises_amplitude_and_lowers_entropy(background_eeg, seizure_like_eeg):
    bg, sz = feats_of(background_eeg), feats_of(seizure_like_eeg)
    for name in ["log_variance", "line_length", "log_peak_to_peak"]:
        i = PER_CHANNEL.index(name)
        assert np.all(sz[:, i] > bg[:, i]), name
    i = PER_CHANNEL.index("spectral_entropy")
    assert np.mean(sz[:, i]) < np.mean(bg[:, i])        # rhythmic = more predictable spectrum
