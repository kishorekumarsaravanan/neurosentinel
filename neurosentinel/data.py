"""Dataset access: seizure annotations and loading of preprocessed recordings."""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path
import numpy as np

from .config import DATA_RAW, CHANNELS
from .edf import read_edf, select_channels
from .preprocess import preprocess


@lru_cache(maxsize=None)
def seizure_table(patient: str) -> dict[str, list[tuple[float, float]]]:
    """Parse <patient>-summary.txt -> {record_name: [(start_s, end_s), ...]}."""
    text = (DATA_RAW / patient / f"{patient}-summary.txt").read_text(errors="ignore")
    table: dict[str, list[tuple[float, float]]] = {}
    for block in re.split(r"(?=File Name:)", text):
        m = re.search(r"File Name:\s*(\S+)\.edf", block)
        if not m:
            continue
        starts = [float(v) for v in re.findall(r"Seizure(?:\s*\d+)?\s*Start Time:\s*(\d+)", block)]
        ends = [float(v) for v in re.findall(r"Seizure(?:\s*\d+)?\s*End Time:\s*(\d+)", block)]
        table[m.group(1)] = list(zip(starts, ends))
    return table


def patient_of(record: str) -> str:
    return record.split("_")[0]


def seizures_of(record: str) -> list[tuple[float, float]]:
    return seizure_table(patient_of(record)).get(record, [])


def edf_path(record: str) -> Path:
    return DATA_RAW / patient_of(record) / f"{record}.edf"


def available_records() -> list[str]:
    return sorted(p.stem for p in DATA_RAW.glob("chb*/*.edf"))


def load_raw(record: str) -> np.ndarray:
    """Raw (18, n) microvolt array for a record, montage-selected."""
    return select_channels(read_edf(edf_path(record)), CHANNELS)


def load_record(record: str) -> np.ndarray:
    """Filtered (18, n) float32 array for a record."""
    return preprocess(load_raw(record))


def seizure_mask(n_samples: int, fs: float, seizures) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for s, e in seizures:
        mask[int(s * fs): int(e * fs)] = True
    return mask
