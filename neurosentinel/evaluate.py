"""Window-level and event-level (clinical) evaluation."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix)

from .config import ALARM_K, ALARM_N, ALARM_THRESHOLD, ALARM_REFRACTORY_SEC


def window_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    out = {
        "accuracy": (tp + tn) / len(y), "precision": prec, "recall": sens, "sensitivity": sens,
        "specificity": spec, "f1": f1, "balanced_accuracy": (sens + spec) / 2,
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_windows": int(len(y)), "n_seizure_windows": int(y.sum()),
    }
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        k = max(1, len(fpr) // 300)
        out["roc"] = {"fpr": fpr[::k].round(4).tolist() + [1.0], "tpr": tpr[::k].round(4).tolist() + [1.0]}
        k = max(1, len(pr) // 300)
        out["pr"] = {"precision": pr[::k].round(4).tolist(), "recall": rc[::k].round(4).tolist()}
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}


def alarm_signal(p: np.ndarray, thr: float = ALARM_THRESHOLD, k: int = ALARM_K, n: int = ALARM_N) -> np.ndarray:
    """k-of-n smoothing of 1 Hz decisions -> boolean alarm state per step."""
    pos = (np.asarray(p) >= thr).astype(int)
    cs = np.concatenate([[0], np.cumsum(pos)])
    idx = np.arange(len(pos))
    lo = np.maximum(0, idx - n + 1)
    return (cs[idx + 1] - cs[lo]) >= k


def alarm_events(alarm: np.ndarray, times: np.ndarray, refractory: float = ALARM_REFRACTORY_SEC) -> list[float]:
    """Times of distinct alarm onsets (rising edges, merged within the refractory period)."""
    onsets = times[np.flatnonzero(np.diff(np.concatenate([[0], alarm.astype(int)])) == 1)]
    events: list[float] = []
    for t in onsets:
        if not events or t - events[-1] > refractory:
            events.append(float(t))
    return events


def event_metrics(records: list[dict], thr: float = ALARM_THRESHOLD, k: int = ALARM_K, n: int = ALARM_N) -> dict:
    """records: [{record, times, p, seizures, duration}] for one model -> clinical metrics."""
    n_sz = det = fa = 0
    latencies, hours, per_seizure = [], 0.0, []
    for r in records:
        alarm = alarm_signal(r["p"], thr, k, n)
        events = alarm_events(alarm, r["times"])
        hours += r["duration"] / 3600
        for s, e in r["seizures"]:
            n_sz += 1
            hit = r["times"][alarm & (r["times"] >= s) & (r["times"] <= e + 10)]
            ok = len(hit) > 0
            det += ok
            lat = float(hit[0] - s) if ok else None
            if ok:
                latencies.append(lat)
            per_seizure.append({"record": r["record"], "onset": s, "end": e, "detected": bool(ok), "latency": lat})
        for t in events:
            if not any(s - 30 <= t <= e + 60 for s, e in r["seizures"]):
                fa += 1
    return {
        "seizures": n_sz, "detected": int(det), "event_sensitivity": det / n_sz if n_sz else 0.0,
        "mean_latency": float(np.mean(latencies)) if latencies else None,
        "false_alarms": int(fa), "hours": hours, "fa_per_hour": fa / hours if hours else 0.0,
        "per_seizure": per_seizure,
    }
