"""Dataset access: seizure annotations and real-recording checks."""
import numpy as np
import pytest

import neurosentinel.data as data
from neurosentinel.config import CHANNELS, FS, PATIENTS
from conftest import needs_data

SUMMARY = """Data Sampling Rate: 256 Hz

File Name: chb99_01.edf
File Start Time: 10:00:00
File End Time: 11:00:00
Number of Seizures in File: 0

File Name: chb99_02.edf
File Start Time: 11:00:00
File End Time: 12:00:00
Number of Seizures in File: 2
Seizure 1 Start Time: 100 seconds
Seizure 1 End Time: 140 seconds
Seizure 2 Start Time: 900 seconds
Seizure 2 End Time: 960 seconds

File Name: chb99_03.edf
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
"""


@pytest.fixture
def fake_raw(tmp_path, monkeypatch):
    (tmp_path / "chb99").mkdir()
    (tmp_path / "chb99" / "chb99-summary.txt").write_text(SUMMARY)
    monkeypatch.setattr(data, "DATA_RAW", tmp_path)
    data.seizure_table.cache_clear()
    yield tmp_path
    data.seizure_table.cache_clear()


def test_summary_parsing_handles_both_annotation_formats(fake_raw):
    tab = data.seizure_table("chb99")
    assert tab["chb99_01"] == []
    assert tab["chb99_02"] == [(100.0, 140.0), (900.0, 960.0)]       # "Seizure 1 Start Time" format
    assert tab["chb99_03"] == [(2996.0, 3036.0)]                      # "Seizure Start Time" format
    assert data.seizures_of("chb99_02") == tab["chb99_02"]
    assert data.patient_of("chb99_02") == "chb99"


def test_seizure_mask():
    m = data.seizure_mask(10 * FS, FS, [(2.0, 4.0)])
    assert m.sum() == 2 * FS and m[2 * FS] and not m[4 * FS]


def test_train_and_test_recordings_are_disjoint():
    for patient, split in PATIENTS.items():
        assert not set(split["train"]) & set(split["test"]), patient
        assert all(r.startswith(patient) for r in split["train"] + split["test"])


@needs_data
@pytest.mark.data
def test_real_recording_loads_with_montage_and_annotations():
    data.seizure_table.cache_clear()
    x = data.load_record("chb01_03")
    assert x.shape == (len(CHANNELS), 3600 * FS) and x.dtype == np.float32
    assert data.seizures_of("chb01_03") == [(2996.0, 3036.0)]
    assert np.isfinite(x).all() and 1 < np.std(x) < 500           # plausible scalp EEG amplitude in uV
