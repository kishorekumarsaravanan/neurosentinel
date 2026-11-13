"""Signal conditioning and segmentation."""
from __future__ import annotations
import numpy as np
from scipy import signal

from .config import FS, BANDPASS, NOTCH_HZ, WIN


def bandpass(x: np.ndarray, lo: float, hi: float, fs: float = FS, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def notch(x: np.ndarray, f0: float = NOTCH_HZ, fs: float = FS, q: float = 30.0) -> np.ndarray:
    b, a = signal.iirnotch(f0, q, fs=fs)
    return signal.filtfilt(b, a, x, axis=-1).astype(np.float32)


def preprocess(x: np.ndarray, fs: float = FS, band=BANDPASS, notch_hz=NOTCH_HZ) -> np.ndarray:
    if notch_hz:
        x = notch(x, notch_hz, fs)
    return bandpass(x, band[0], band[1], fs)


def window_starts(n_samples: int, stride: int, win: int = WIN) -> np.ndarray:
    return np.arange(0, n_samples - win + 1, stride)


def make_windows(x: np.ndarray, starts: np.ndarray, win: int = WIN) -> np.ndarray:
    """(C, N) -> (len(starts), C, win) view-free copy."""
    idx = starts[:, None] + np.arange(win)[None, :]
    return x[:, idx].transpose(1, 0, 2).astype(np.float32, copy=False)


def seizure_overlap(starts: np.ndarray, mask: np.ndarray, win: int = WIN) -> np.ndarray:
    """Fraction of each window that lies inside an annotated seizure."""
    cs = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    return (cs[starts + win] - cs[starts]) / win
