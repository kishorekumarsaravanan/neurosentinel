"""The three competing detectors: SVM (RBF), Random Forest and a 1D-CNN."""
from __future__ import annotations
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CHANNELS, MODELS_DIR


def make_svm():
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                             probability=True, random_state=0))


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=0)


class SeizureCNN(nn.Module):
    """Compact 1D-CNN that learns its own features from the filtered multichannel EEG."""

    def __init__(self, n_channels: int = len(CHANNELS)):
        super().__init__()

        def block(cin, cout, k, pool):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
                                 nn.ReLU(), nn.MaxPool1d(pool) if pool else nn.Identity())

        self.features = nn.Sequential(
            block(n_channels, 32, 7, 4),   # 1024 -> 256
            block(32, 64, 7, 4),           # 256 -> 64
            block(64, 128, 5, 4),          # 64 -> 16
            block(128, 128, 3, 0),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class CNNDetector:
    """Training / inference wrapper with per-channel amplitude normalisation."""

    def __init__(self, scale: np.ndarray | None = None, state=None):
        self.net = SeizureCNN()
        self.scale = scale
        if state is not None:
            self.net.load_state_dict(state)
        self.history: list[dict] = []

    def _norm(self, W: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.clip(W / self.scale[None, :, None], -25, 25).astype(np.float32))

    def fit(self, W: np.ndarray, y: np.ndarray, epochs: int = 35, lr: float = 1e-3, seed: int = 0):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.scale = (W[y == 0].std(axis=(0, 2)) + 1e-6).astype(np.float32)
        X = self._norm(W)
        Y = torch.from_numpy(y.astype(np.float32))
        idx = rng.permutation(len(X))
        n_val = max(1, int(0.15 * len(X)))
        val, tr = idx[:n_val], idx[n_val:]
        pos_w = torch.tensor((Y[tr] == 0).sum().item() / max(1, (Y[tr] == 1).sum().item()))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, best_state = np.inf, None
        for ep in range(epochs):
            self.net.train()
            perm = torch.from_numpy(rng.permutation(tr))
            tot = 0.0
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]
                xb = X[b]
                # augmentation: amplitude scaling, additive noise, random circular time shift
                xb = xb * (0.8 + 0.4 * torch.rand(len(b), 1, 1)) + 0.05 * torch.randn_like(xb)
                xb = torch.roll(xb, int(rng.integers(-128, 128)), dims=-1)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), Y[b])
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            sched.step()
            self.net.eval()
            with torch.no_grad():
                logits = self.net(X[val])
                vl = loss_fn(logits, Y[val]).item()
                vacc = ((logits > 0).float() == Y[val]).float().mean().item()
            self.history.append({"epoch": ep + 1, "train_loss": tot / len(tr), "val_loss": vl, "val_acc": vacc})
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, W: np.ndarray, batch: int = 256) -> np.ndarray:
        self.net.eval()
        out = [torch.sigmoid(self.net(self._norm(W[i:i + batch]))).numpy() for i in range(0, len(W), batch)]
        return np.concatenate(out) if out else np.zeros(0)

    def saliency(self, W: np.ndarray) -> np.ndarray:
        """|gradient x input| map of shape (C, T) for a single window W (C, T)."""
        self.net.eval()
        x = self._norm(W[None]).requires_grad_(True)
        self.net(x).sum().backward()
        return (x.grad * x).abs()[0].detach().numpy()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def save(self, path: Path):
        torch.save({"state": self.net.state_dict(), "scale": self.scale, "history": self.history}, path)

    @classmethod
    def load(cls, path: Path) -> "CNNDetector":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(scale=ck["scale"], state=ck["state"])
        obj.history = ck.get("history", [])
        return obj


class PatientModels:
    """All three trained detectors for one patient."""

    def __init__(self, patient: str):
        d = MODELS_DIR / patient
        self.patient = patient
        self.svm = joblib.load(d / "svm.joblib")
        self.rf = joblib.load(d / "rf.joblib")
        self.cnn = CNNDetector.load(d / "cnn.pt")

    def predict_all(self, W: np.ndarray, F: np.ndarray) -> dict[str, np.ndarray]:
        return {"svm": self.svm.predict_proba(F)[:, 1],
                "rf": self.rf.predict_proba(F)[:, 1],
                "cnn": self.cnn.predict_proba(W)}


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0
