"""Central configuration: paths, montage, signal parameters and patient splits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
PRED_DIR = ARTIFACTS / "predictions"
DEMO_DIR = ARTIFACTS / "demo"

FS = 256                      # CHB-MIT sampling rate (Hz)
WIN_SEC = 4                   # analysis window length (s)
WIN = FS * WIN_SEC            # 1024 samples
TEST_STRIDE_SEC = 1           # stride used for streaming predictions
TRAIN_SEIZURE_STRIDE_SEC = 1  # overlap-augmentation for the rare seizure class
NONSEIZURE_RATIO = 8          # non-seizure : seizure windows kept for training

BANDPASS = (0.5, 40.0)
NOTCH_HZ = 60.0               # recordings were made in the US (60 Hz mains)

# 18 bipolar channels of the longitudinal "double banana" montage (common to all patients)
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

# Patient-specific protocol: training and test recordings are disjoint;
# every test set holds unseen seizures plus a fully seizure-free hour.
PATIENTS = {
    "chb01": {
        "train": ["chb01_01", "chb01_02", "chb01_03", "chb01_04", "chb01_05",
                  "chb01_15", "chb01_16", "chb01_18"],
        "test": ["chb01_06", "chb01_21", "chb01_26"],
    },
    "chb03": {
        "train": ["chb03_01", "chb03_02", "chb03_03", "chb03_04", "chb03_05",
                  "chb03_06", "chb03_34"],
        "test": ["chb03_07", "chb03_35", "chb03_36"],
    },
}

MODEL_NAMES = {"svm": "SVM (RBF)", "rf": "Random Forest", "cnn": "1D-CNN"}
# IBM Carbon categorical palette, positions 1-3 (purple 70, cyan 50, teal 70)
MODEL_COLORS = {"svm": "#6929C4", "rf": "#005D5D", "cnn": "#1192E8"}

# Event-level alarm logic: raise an alarm when K of the last N 1-s predictions are positive
ALARM_K, ALARM_N = 3, 4
ALARM_THRESHOLD = 0.5
ALARM_REFRACTORY_SEC = 60
