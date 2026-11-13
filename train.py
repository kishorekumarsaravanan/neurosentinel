"""End-to-end NeuroSentinel pipeline.

    python train.py            # train + evaluate + export app assets for all patients

Steps per patient (patient-specific protocol, train/test split by recording):
  1. load EDF -> 18-channel bipolar montage -> 60 Hz notch + 0.5-40 Hz band-pass
  2. segment into 4 s windows (seizure windows with 75 % overlap as augmentation)
  3. undersample non-seizure windows (class imbalance), extract 378 hand-crafted features (21 per channel)
  4. train SVM (RBF), Random Forest (features) and 1D-CNN (filtered raw EEG)
  5. stream every unseen test hour at 1 s stride -> window + event-level metrics
  6. export predictions, demo clips and game rounds for the Streamlit app
"""
from __future__ import annotations
import base64
import gc
import json
import time
import joblib
import numpy as np

from neurosentinel.config import (FS, WIN, PATIENTS, NONSEIZURE_RATIO, MODELS_DIR, PRED_DIR, DEMO_DIR,
                                  ARTIFACTS, CHANNELS, MODEL_NAMES, BANDS)
from neurosentinel.data import load_record, seizures_of, seizure_mask, edf_path
from neurosentinel.preprocess import window_starts, make_windows, seizure_overlap
from neurosentinel.features import extract_features, FEATURE_NAMES, PER_CHANNEL
from neurosentinel.models import make_svm, make_rf, CNNDetector
from neurosentinel.evaluate import window_metrics, event_metrics, alarm_signal

RNG = np.random.default_rng(42)
DISPLAY_FS = 128
SCALE_Q = 10.0          # int16 quantisation: 0.1 uV resolution


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def record_duration(rec: str) -> float:
    head = edf_path(rec).read_bytes()[:256].decode("latin-1")
    return int(head[236:244]) * float(head[244:252])


# --------------------------------------------------------------------------- training data
def build_train_set(records: list[str]):
    sz_secs = sum(max(0.0, e - s - 3) for r in records for s, e in seizures_of(r))
    ns_total = sum(record_duration(r) / 4 for r in records)
    keep_frac = min(1.0, NONSEIZURE_RATIO * sz_secs / ns_total)
    Ws, ys, srcs = [], [], []
    for rec in records:
        x = load_record(rec)
        n = x.shape[1]
        mask = seizure_mask(n, FS, seizures_of(rec))
        st1 = window_starts(n, FS)                 # 1 s stride -> overlap augmentation for seizures
        sz = st1[seizure_overlap(st1, mask) >= 0.9]
        st4 = window_starts(n, WIN)                # non-overlapping background windows
        ns = st4[seizure_overlap(st4, mask) == 0]
        ns = ns[RNG.random(len(ns)) < keep_frac]
        for starts, lab in ((sz, 1), (ns, 0)):
            if len(starts):
                Ws.append(make_windows(x, starts))
                ys.append(np.full(len(starts), lab))
                srcs += [rec] * len(starts)
        log(f"   {rec}: {len(sz)} seizure / {len(ns)} background windows")
    return np.concatenate(Ws), np.concatenate(ys)


# --------------------------------------------------------------------------- streaming test
def stream_record(rec: str, models: dict) -> dict:
    x = load_record(rec)
    n = x.shape[1]
    seizures = seizures_of(rec)
    mask = seizure_mask(n, FS, seizures)
    st = window_starts(n, FS)
    y = (seizure_overlap(st, mask) >= 0.5).astype(int)
    times = (st + WIN) / FS                      # decision available at the end of each window
    probs = {k: [] for k in models}
    feats_all, timing = [], {k: 0.0 for k in models}
    for i in range(0, len(st), 300):
        W = make_windows(x, st[i:i + 300])
        t0 = time.perf_counter()
        F = extract_features(W)
        t_feat = time.perf_counter() - t0
        feats_all.append(F[::4])
        for k, m in models.items():
            t0 = time.perf_counter()
            p = m.predict_proba(W) if k == "cnn" else m.predict_proba(F)[:, 1]
            timing[k] += time.perf_counter() - t0 + (0 if k == "cnn" else t_feat)
            probs[k].append(p)
    probs = {k: np.concatenate(v).astype(np.float32) for k, v in probs.items()}
    F4 = np.concatenate(feats_all).reshape(-1, len(CHANNELS), len(PER_CHANNEL))
    i0 = PER_CHANNEL.index("rel_power_delta")
    return {"record": rec, "x": x, "times": times, "y": y, "probs": probs, "seizures": seizures,
            "duration": n / FS, "bandpow": F4[..., i0:i0 + len(BANDS)].astype(np.float16),
            "linelen": F4[..., PER_CHANNEL.index("line_length")].astype(np.float16),
            "ms_per_window": {k: 1000 * v / len(st) for k, v in timing.items()}}


# --------------------------------------------------------------------------- app exports
def encode_eeg(x: np.ndarray, t0: float, t1: float) -> str:
    seg = x[:, int(t0 * FS):int(t1 * FS):FS // DISPLAY_FS]
    q = np.clip(np.round(seg * SCALE_Q), -32767, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode()


def clip_probs(res: dict, t0: float, t1: float) -> dict:
    sel = (res["times"] >= t0) & (res["times"] <= t1)
    return {"t": (res["times"][sel] - t0).round(2).tolist(),
            **{k: v[sel].round(3).tolist() for k, v in res["probs"].items()}}


def make_clip(res: dict, t0: float, t1: float, title: str, kind: str) -> dict:
    t0, t1 = max(0.0, t0), min(res["duration"], t1)
    return {
        "id": f"{res['record']}_{int(t0)}", "title": title, "kind": kind, "record": res["record"],
        "patient": res["record"].split("_")[0], "t0": t0, "duration": t1 - t0, "fs": DISPLAY_FS,
        "scale": SCALE_Q, "channels": CHANNELS, "eeg": encode_eeg(res["x"], t0, t1),
        "seizures": [[s - t0, e - t0] for s, e in res["seizures"] if e > t0 and s < t1],
        "probs": clip_probs(res, t0, t1),
    }


def first_alarm(res: dict, model: str, t0: float, t1: float):
    alarm = alarm_signal(res["probs"][model])
    hit = res["times"][alarm & (res["times"] >= t0) & (res["times"] <= t1)]
    return float(hit[0] - t0) if len(hit) else None


def make_assets(res: dict) -> tuple[list[dict], list[dict]]:
    """Monitor clips + game rounds for one streamed test record (needs res['x'])."""
    rec, clips, rounds = res["record"], [], []
    if res["seizures"]:
        for j, (s, e) in enumerate(res["seizures"]):
            clips.append(make_clip(res, s - 150, e + 60, f"{rec} · seizure {j + 1}", "seizure"))
            t0 = s - float(RNG.uniform(18, 38))
            rnd = make_clip(res, t0, t0 + 60, "", "seizure")
            rnd["ai"] = {m: first_alarm(res, m, t0, t0 + 60) for m in res["probs"]}
            rounds.append(rnd)
    else:
        clips.append(make_clip(res, 1500, 1740, f"{rec} · seizure-free hour", "normal"))
        t0 = float(RNG.uniform(600, 2800))
        rnd = make_clip(res, t0, t0 + 60, "", "normal")
        rnd["ai"] = {m: first_alarm(res, m, t0, t0 + 60) for m in res["probs"]}
        rounds.append(rnd)
    return clips, rounds


def write_app_assets(clips: list[dict], rounds: list[dict]):
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    rounds = [rounds[i] for i in RNG.permutation(len(rounds))]
    for i, r in enumerate(rounds):
        r["title"] = f"Round {i + 1}"
    (DEMO_DIR / "monitor_clips.json").write_text(json.dumps(clips))
    (DEMO_DIR / "game_rounds.json").write_text(json.dumps(rounds))
    log(f"exported {len(clips)} monitor clips and {len(rounds)} game rounds")


# --------------------------------------------------------------------------- main
def main():
    for d in (MODELS_DIR, PRED_DIR, DEMO_DIR):
        d.mkdir(parents=True, exist_ok=True)
    results = {"patients": {}, "models": MODEL_NAMES, "feature_names": FEATURE_NAMES,
               "created": time.strftime("%Y-%m-%d %H:%M")}
    all_tests, pooled = [], {k: {"y": [], "p": []} for k in MODEL_NAMES}
    all_clips, all_rounds = [], []

    for patient, split in PATIENTS.items():
        log(f"=== {patient}: building training set")
        W, y = build_train_set(split["train"])
        log(f"   train windows: {len(y)} ({int(y.sum())} seizure)")
        t0 = time.perf_counter(); F = extract_features(W); t_feat = time.perf_counter() - t0

        out_dir = MODELS_DIR / patient
        out_dir.mkdir(parents=True, exist_ok=True)
        info = {"train_windows": int(len(y)), "train_seizure_windows": int(y.sum()),
                "train_records": split["train"], "test_records": split["test"], "models": {}}

        log("   training SVM"); t0 = time.perf_counter()
        svm = make_svm().fit(F, y); t_svm = time.perf_counter() - t0
        log("   training Random Forest"); t0 = time.perf_counter()
        rf = make_rf().fit(F, y); t_rf = time.perf_counter() - t0
        log("   training 1D-CNN"); t0 = time.perf_counter()
        cnn = CNNDetector().fit(W, y); t_cnn = time.perf_counter() - t0
        joblib.dump(svm, out_dir / "svm.joblib"); joblib.dump(rf, out_dir / "rf.joblib")
        cnn.save(out_dir / "cnn.pt")
        np.save(out_dir / "background_feature_median.npy", np.median(F[y == 0], axis=0))
        del W, F
        gc.collect()

        models = {"svm": svm, "rf": rf, "cnn": cnn}
        train_time = {"svm": t_svm + t_feat, "rf": t_rf + t_feat, "cnn": t_cnn}
        size = {"svm": int(svm[-1].n_support_.sum()), "rf": int(sum(t.tree_.node_count for t in rf.estimators_)),
                "cnn": cnn.n_params()}

        tests = []
        for rec in split["test"]:
            log(f"   streaming unseen record {rec}")
            res = stream_record(rec, models)
            np.savez_compressed(PRED_DIR / f"{rec}.npz", times=res["times"], y=res["y"],
                                bandpow=res["bandpow"], linelen=res["linelen"],
                                seizures=np.array(res["seizures"]).reshape(-1, 2),
                                **{f"p_{k}": v for k, v in res["probs"].items()})
            c, r = make_assets(res)
            all_clips += c; all_rounds += r
            del res["x"]                      # keep memory flat: the filtered hour is no longer needed
            gc.collect()
            tests.append(res)

        for k in models:
            yy = np.concatenate([r["y"][::4] for r in tests])
            pp = np.concatenate([r["probs"][k][::4] for r in tests])
            pooled[k]["y"].append(yy); pooled[k]["p"].append(pp)
            wm = window_metrics(yy, pp)
            em = event_metrics([{"record": r["record"], "times": r["times"], "p": r["probs"][k],
                                 "seizures": r["seizures"], "duration": r["duration"]} for r in tests])
            info["models"][k] = {"window": wm, "event": em, "train_time_s": train_time[k],
                                 "ms_per_window": float(np.mean([r["ms_per_window"][k] for r in tests])),
                                 "size": size[k]}
            log(f"   {k:4s} sens={wm['sensitivity']:.3f} spec={wm['specificity']:.3f} f1={wm['f1']:.3f} "
                f"auc={wm.get('roc_auc', 0):.3f} | events {em['detected']}/{em['seizures']} "
                f"lat={em['mean_latency']} FA/h={em['fa_per_hour']:.2f}")
        info["cnn_history"] = cnn.history
        imp = rf.feature_importances_
        info["rf_importance"] = imp.round(6).tolist()
        results["patients"][patient] = info
        all_tests += tests

    results["overall"] = {}
    for k in MODEL_NAMES:
        yy = np.concatenate(pooled[k]["y"]); pp = np.concatenate(pooled[k]["p"])
        em = event_metrics([{"record": r["record"], "times": r["times"], "p": r["probs"][k],
                             "seizures": r["seizures"], "duration": r["duration"]} for r in all_tests])
        results["overall"][k] = {"window": window_metrics(yy, pp), "event": em}
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=1))
    write_app_assets(all_clips, all_rounds)
    log("done -> artifacts/")


if __name__ == "__main__":
    main()
