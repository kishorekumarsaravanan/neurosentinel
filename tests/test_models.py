"""The three detectors: construction, training on a separable toy problem, persistence."""
import numpy as np
import pytest
import torch

from neurosentinel.config import CHANNELS, FS, WIN
from neurosentinel.features import extract_features
from neurosentinel.models import SeizureCNN, CNNDetector, make_svm, make_rf


@pytest.fixture(scope="module")
def toy_windows():
    """60 background windows and 20 seizure-like windows (large rhythmic 3 Hz activity)."""
    rng = np.random.default_rng(1)
    t = np.arange(WIN) / FS
    bg = rng.normal(0, 15, (60, len(CHANNELS), WIN))
    sz = 120 * np.sin(2 * np.pi * 3 * t + rng.uniform(0, 6, (20, 1, 1))) + rng.normal(0, 15, (20, len(CHANNELS), WIN))
    W = np.concatenate([bg, sz]).astype(np.float32)
    y = np.array([0] * 60 + [1] * 20)
    return W, y


def test_cnn_architecture_shapes():
    net = SeizureCNN()
    out = net(torch.zeros(4, len(CHANNELS), WIN))
    assert out.shape == (4,)
    assert 100_000 < sum(p.numel() for p in net.parameters()) < 120_000


def test_feature_models_learn_separable_problem(toy_windows):
    W, y = toy_windows
    F = extract_features(W)
    for model in (make_svm(), make_rf()):
        model.fit(F, y)
        p = model.predict_proba(F)[:, 1]
        assert p.shape == (len(y),) and np.all((p >= 0) & (p <= 1))
        assert ((p >= 0.5) == y).mean() > 0.95


def test_cnn_trains_predicts_saves_and_loads(toy_windows, tmp_path):
    W, y = toy_windows
    det = CNNDetector().fit(W, y, epochs=8, seed=0)
    assert len(det.history) == 8 and det.scale.shape == (len(CHANNELS),)
    p = det.predict_proba(W)
    assert p.shape == (len(y),) and np.all((p >= 0) & (p <= 1))
    assert ((p >= 0.5) == y).mean() > 0.9
    det.save(tmp_path / "cnn.pt")
    again = CNNDetector.load(tmp_path / "cnn.pt")
    np.testing.assert_allclose(again.predict_proba(W), p, atol=1e-6)


def test_cnn_saliency_map_shape(toy_windows):
    W, y = toy_windows
    det = CNNDetector(scale=np.ones(len(CHANNELS), np.float32))
    sal = det.saliency(W[-1])
    assert sal.shape == (len(CHANNELS), WIN) and np.all(sal >= 0)
