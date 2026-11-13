"""Hand-crafted EEG features (time- and frequency-domain) per channel."""
from __future__ import annotations
import numpy as np

from .config import FS, BANDS, CHANNELS

TIME_FEATURES = ["log_variance", "skewness", "kurtosis", "line_length", "zero_cross_rate",
                 "hjorth_mobility", "hjorth_complexity", "log_peak_to_peak"]
FREQ_FEATURES = ([f"log_power_{b}" for b in BANDS] + [f"rel_power_{b}" for b in BANDS]
                 + ["spectral_entropy", "peak_frequency", "spectral_edge_90"])
PER_CHANNEL = TIME_FEATURES + FREQ_FEATURES
FEATURE_NAMES = [f"{ch}|{f}" for ch in CHANNELS for f in PER_CHANNEL]

FEATURE_FAMILY = {f: ("time" if f in TIME_FEATURES else "frequency") for f in PER_CHANNEL}


def _features_chunk(W: np.ndarray, fs: float) -> np.ndarray:
    n, c, t = W.shape
    eps = 1e-10
    d = W - W.mean(-1, keepdims=True)
    var = (d ** 2).mean(-1) + eps
    skew = (d ** 3).mean(-1) / var ** 1.5
    kurt = (d ** 4).mean(-1) / var ** 2
    dx = np.diff(d, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var_dx = dx.var(-1) + eps
    var_ddx = ddx.var(-1) + eps
    mobility = np.sqrt(var_dx / var)
    complexity = np.sqrt(var_ddx / var_dx) / mobility
    line_length = np.log(np.abs(dx).mean(-1) + eps)
    zcr = (np.diff(np.signbit(d), axis=-1)).sum(-1) / (t / fs)
    ptp = np.log(d.max(-1) - d.min(-1) + eps)

    spec = np.abs(np.fft.rfft(d * np.hanning(t).astype(np.float32), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(t, 1 / fs)
    band_sel = (freqs >= 0.5) & (freqs <= 40)
    sp = spec[..., band_sel]
    f_sel = freqs[band_sel]
    total = sp.sum(-1) + eps
    abs_pow, rel_pow = [], []
    for lo, hi in BANDS.values():
        m = (f_sel >= lo) & (f_sel < hi)
        p = sp[..., m].sum(-1)
        abs_pow.append(np.log(p + eps))
        rel_pow.append(p / total)
    pn = sp / total[..., None]
    entropy = -(pn * np.log(pn + eps)).sum(-1) / np.log(sp.shape[-1])
    peak = f_sel[sp.argmax(-1)]
    sef = f_sel[(np.cumsum(pn, -1) < 0.9).sum(-1).clip(max=len(f_sel) - 1)]

    per = [np.log(var), skew, kurt, line_length, zcr, mobility, complexity, ptp,
           *abs_pow, *rel_pow, entropy, peak, sef]
    F = np.stack(per, axis=-1)          # (n, c, n_feat)
    return F.reshape(n, -1).astype(np.float32)


def extract_features(W: np.ndarray, fs: float = FS, chunk: int = 512) -> np.ndarray:
    """W: (n_windows, n_channels, n_samples) -> (n_windows, n_channels * n_features)."""
    if len(W) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    return np.concatenate([_features_chunk(W[i:i + chunk], fs) for i in range(0, len(W), chunk)])


def band_powers(W: np.ndarray, fs: float = FS) -> np.ndarray:
    """Relative band power per channel: (n, c, n_bands)."""
    F = extract_features(W, fs).reshape(len(W), len(CHANNELS), len(PER_CHANNEL))
    i0 = PER_CHANNEL.index("rel_power_delta")
    return F[..., i0:i0 + len(BANDS)]
