import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from nsapp import ui
from neurosentinel.evaluate import window_metrics, event_metrics

ui.hero("Evaluate / Model Comparison",
        "Three approaches, <em>one task</em>",
        "A <b>kernel machine</b> that draws a non-linear boundary through feature space, an <b>ensemble</b> of 300 "
        "decision trees, and a <b>convolutional network</b> that learns its own features from the signal. All are "
        "tested on held-out hours, with window-level and clinical event-level metrics.",
        index="07", meta="TEST DATA <b>6 held-out hours</b><br>SEIZURES <b>4</b><br>WINDOWS <b>4 s, non-overlapping</b>")
ui.need_artifacts()

R = ui.results()
KEYS = ["svm", "rf", "cnn"]
patients = list(R["patients"])
scope = st.segmented_control("Scope", ["All patients"] + patients, default="All patients") or "All patients"


def M(k):
    if scope == "All patients":
        o = R["overall"][k]
        per = [R["patients"][p]["models"][k] for p in patients]
        return {"window": o["window"], "event": o["event"],
                "ms": float(np.mean([x["ms_per_window"] for x in per])), "size": per[0]["size"],
                "train": float(np.sum([x["train_time_s"] for x in per]))}
    x = R["patients"][scope]["models"][k]
    return {"window": x["window"], "event": x["event"], "ms": x["ms_per_window"], "size": x["size"], "train": x["train_time_s"]}


S = {k: M(k) for k in KEYS}

# ------------------------------------------------------------------ model cards
badges = {}
best = lambda f, rev=True: (max if rev else min)(KEYS, key=f)
badges.setdefault(best(lambda k: S[k]["window"]["f1"]), "Best F1")
lat = {k: S[k]["event"]["mean_latency"] if S[k]["event"]["mean_latency"] is not None else 1e9 for k in KEYS}
badges.setdefault(best(lambda k: lat[k], rev=False), "Fastest detection")
badges.setdefault(best(lambda k: S[k]["event"]["fa_per_hour"], rev=False), "Fewest false alarms")
badges.setdefault(best(lambda k: S[k]["window"].get("roc_auc", 0)), "Highest ROC-AUC")
badges.setdefault(best(lambda k: S[k]["ms"], rev=False), "Fastest inference")
CLASS = {"svm": "Traditional ML · RBF kernel", "rf": "Ensemble · 300 trees", "cnn": "Deep learning · 4 conv blocks"}
NOTE = {"svm": "Finds a maximum-margin boundary in the 378-dimensional feature space.",
        "rf": "Averages the votes of 300 decorrelated decision trees.",
        "cnn": "Learns its own filters directly from the multichannel signal."}
SIZE_LABEL = {"svm": "support vectors", "rf": "tree nodes", "cnn": "parameters"}

cols = st.columns(3, gap="medium")
for col, k in zip(cols, KEYS):
    w, e, c = S[k]["window"], S[k]["event"], ui.MODEL_COLORS[k]
    rows = [("Sensitivity", w["sensitivity"]), ("Specificity", w["specificity"]), ("Precision", w["precision"]),
            ("F1-score", w["f1"]), ("ROC-AUC", w.get("roc_auc", 0)), ("Seizures detected", e["event_sensitivity"])]
    stats = "".join(f'<div class="stat"><span>{n}</span><b>{v * 100:.1f}%</b><div class="bar"><i style="width:{v * 100:.1f}%"></i></div></div>' for n, v in rows)
    latency = f"{e['mean_latency']:.1f} s" if e["mean_latency"] is not None else "n/a"
    badge = f'<div class="badge">{badges[k]}</div>' if k in badges else ""
    col.markdown(
        f"""<div class="ns-model" style="--acc:{c}">{badge}
        <div class="head"><div class="ico">{ui.icon(ui.ICON[k], 20, c)}</div><div><div class="nm">{ui.MODEL_NAMES[k]}</div><div class="cls">{CLASS[k]}</div></div></div>
        {stats}
        <div class="kpis">
          <div class="kr"><span>Mean detection latency</span><b>{latency}</b></div>
          <div class="kr"><span>False alarms per hour</span><b>{e['fa_per_hour']:.2f}</b></div>
          <div class="kr"><span>Inference per window</span><b>{S[k]['ms']:.2f} ms</b></div>
        </div>
        <div class="foot">{e['detected']}/{e['seizures']} seizures · {S[k]['size']:,} {SIZE_LABEL[k]} · trained in {S[k]['train']:.0f} s</div>
        <div class="quip">{NOTE[k]}</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ radar + table
c1, c2 = st.columns([1, 1.15], gap="medium")
with c1:
    axes = ["Sensitivity", "Specificity", "Precision", "F1", "ROC-AUC", "PR-AUC", "Event sens."]
    fig = go.Figure()
    allv = []
    for k in KEYS:
        w, e = S[k]["window"], S[k]["event"]
        vals = [w["sensitivity"], w["specificity"], w["precision"], w["f1"], w.get("roc_auc", 0), w.get("pr_auc", 0), e["event_sensitivity"]]
        allv += vals
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=axes + axes[:1], name=ui.MODEL_NAMES[k], fill="toself",
                                      line=dict(color=ui.MODEL_COLORS[k], width=2), fillcolor=ui.hexa(ui.MODEL_COLORS[k], 0.08),
                                      marker=dict(size=5)))
    lo = max(0.0, np.floor((min(allv) - 0.05) * 20) / 20)     # zoom the radial axis so differences are visible
    fig.update_layout(polar=dict(radialaxis=dict(range=[lo, 1], gridcolor=ui.GRID_STRONG, linecolor=ui.LINE, tickfont=dict(size=9, family="IBM Plex Mono"), tickformat=".0%"),
                                 angularaxis=dict(gridcolor=ui.GRID_STRONG, linecolor=ui.LINE_STRONG, tickfont=dict(size=11.5, color=ui.INK))),
                      title="Performance profile")
    fig = ui.style(fig, 450)
    fig.update_layout(legend=dict(y=-0.12, yanchor="top", x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True, theme=None)
with c2:
    tbl = pd.DataFrame({
        ui.MODEL_NAMES[k]: {
            "Accuracy": S[k]["window"]["accuracy"], "Precision": S[k]["window"]["precision"],
            "Recall (Sensitivity)": S[k]["window"]["recall"], "Specificity": S[k]["window"]["specificity"],
            "F1-score": S[k]["window"]["f1"], "Balanced accuracy": S[k]["window"]["balanced_accuracy"],
            "ROC-AUC": S[k]["window"].get("roc_auc", np.nan), "PR-AUC": S[k]["window"].get("pr_auc", np.nan),
            "Seizures detected": S[k]["event"]["event_sensitivity"],
            "Mean latency (s)": S[k]["event"]["mean_latency"], "False alarms / hour": S[k]["event"]["fa_per_hour"],
        } for k in KEYS})
    pct = ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-score", "Balanced accuracy", "ROC-AUC", "PR-AUC", "Seizures detected"]
    fmt = tbl.copy().astype(object)
    for r in tbl.index:
        fmt.loc[r] = [("n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v * 100:.2f} %" if r in pct else f"{v:.2f}")) for v in tbl.loc[r]]
    st.markdown("##### Scorecard")
    st.dataframe(fmt, use_container_width=True, height=422)

# ------------------------------------------------------------------ detail tabs
tabs = st.tabs([":material/grid_on: Confusion matrices", ":material/show_chart: ROC and PR curves", ":material/timer: Detection latency",
                ":material/balance: Accuracy paradox", ":material/tune: Alarm tuning", ":material/model_training: CNN training"])

with tabs[0]:
    fig = make_subplots(rows=1, cols=3, subplot_titles=[ui.MODEL_NAMES[k] for k in KEYS], horizontal_spacing=0.05)
    for i, k in enumerate(KEYS):
        cm = np.array(S[k]["window"]["confusion"])
        norm = cm / cm.sum(1, keepdims=True)
        hi_txt, lo_txt = ui.PAPER, ui.INK   # readable on saturated vs. pale cells in both themes
        text = [[f"<span style='color:{hi_txt if norm[r][c] > 0.5 else lo_txt}'><b>{cm[r][c]:,}</b><br>{norm[r][c] * 100:.1f}%</span>" for c in range(2)] for r in range(2)]
        fig.add_trace(go.Heatmap(z=norm, x=["Pred: normal", "Pred: seizure"], y=["True: normal", "True: seizure"], text=text,
                                 texttemplate="%{text}", textfont=dict(size=15, family="IBM Plex Mono"),
                                 colorscale=[[0, ui.SURFACE], [1, ui.MODEL_COLORS[k]]], showscale=False,
                                 zmin=0, zmax=1, hovertemplate="%{y}<br>%{x}<br>%{z:.3f}<extra></extra>"), 1, i + 1)
        fig.update_yaxes(autorange="reversed", showticklabels=(i == 0), showgrid=False, row=1, col=i + 1)
        fig.update_xaxes(showgrid=False, row=1, col=i + 1)
    fig.update_annotations(font=dict(family="Newsreader", size=17, color=ui.INK))
    st.plotly_chart(ui.style(fig, 360, legend=False), use_container_width=True, theme=None)
    st.caption("Rows are normalised: the diagonal shows specificity (top-left) and sensitivity (bottom-right). "
               "Counts are non-overlapping 4-s windows from the held-out hours.")

with tabs[1]:
    c1, c2 = st.columns(2)
    f1 = go.Figure(); f2 = go.Figure()
    for k in KEYS:
        w = S[k]["window"]
        if "roc" in w:
            f1.add_trace(go.Scatter(x=w["roc"]["fpr"], y=w["roc"]["tpr"], name=f"{ui.MODEL_NAMES[k]} · AUC {w['roc_auc']:.3f}", line=dict(color=ui.MODEL_COLORS[k], width=2)))
            f2.add_trace(go.Scatter(x=w["pr"]["recall"], y=w["pr"]["precision"], name=f"{ui.MODEL_NAMES[k]} · AP {w['pr_auc']:.3f}", line=dict(color=ui.MODEL_COLORS[k], width=2)))
    f1.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=ui.INK3, dash="dot", width=1), showlegend=False))
    f1.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    f2.update_layout(title="Precision–recall curve", xaxis_title="Recall", yaxis_title="Precision")
    f1 = ui.style(f1, 420)
    f1.update_layout(legend=dict(y=-0.22, yanchor="top", x=0.5, xanchor="center"))
    f2 = ui.style(f2, 420)
    f2.update_layout(legend=dict(y=-0.22, yanchor="top", x=0.5, xanchor="center"))
    c1.plotly_chart(f1, use_container_width=True, theme=None)
    c2.plotly_chart(f2, use_container_width=True, theme=None)
    st.caption("With fewer than 1 % ictal windows, the precision–recall curve is the more informative of the two.")

with tabs[2]:
    rows = []
    for k in KEYS:
        for s in S[k]["event"]["per_seizure"]:
            rows.append({"model": k, "seizure": f"{s['record']} @ {int(s['onset'])} s", "latency": s["latency"], "dur": s["end"] - s["onset"]})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for k in KEYS:
        d = df[df.model == k]
        fig.add_trace(go.Bar(y=d.seizure, x=d.latency.fillna(0), orientation="h", name=ui.MODEL_NAMES[k],
                             marker=dict(color=ui.MODEL_COLORS[k]), text=[("missed" if pd.isna(v) else f"{v:.0f} s") for v in d.latency],
                             textposition="outside", textfont=dict(family="IBM Plex Mono", size=11)))
    durs = df.drop_duplicates("seizure")
    fig.add_trace(go.Scatter(y=durs.seizure, x=durs.dur, mode="markers", name="seizure end", marker=dict(symbol="line-ns", size=22, line=dict(width=2, color=ui.SEIZURE))))
    fig.update_layout(barmode="group", title="Seconds from annotated onset to first alarm", xaxis_title="seconds after onset")
    fig.update_yaxes(tickfont=dict(family="IBM Plex Mono", size=11, color=ui.INK))
    fig = ui.style(fig, 140 + 70 * len(durs))
    fig.update_layout(legend=dict(y=-0.22, yanchor="top", x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption("The red tick marks the end of each seizure. An alarm before that point still leaves time for an "
               "intervention such as rescue medication, a nurse call or video capture.")

with tabs[3]:
    preds = [ui.prediction(r) for p in patients if scope in ("All patients", p) for r in R["patients"][p]["test_records"]]
    y = np.concatenate([p["y"][::4] for p in preds])
    lazy = window_metrics(y, np.zeros(len(y)))
    c1, c2 = st.columns([1, 1.3], gap="medium")
    with c1:
        st.markdown(f"""<div class="ns-panel"><div class="ns-label">Baseline: always predict "no seizure"</div>
        <div style="font-family:IBM Plex Mono;font-size:56px;color:var(--ink);line-height:1.1;margin-top:14px">{lazy['accuracy'] * 100:.2f}%</div>
        <div style="color:var(--ink2);font-size:15px">accuracy, with <b style="color:var(--seizure)">zero</b> seizures detected.</div>
        <div style="color:var(--ink2);margin-top:14px;font-size:14px;line-height:1.6">Only {y.mean() * 100:.2f} % of test windows are ictal, so accuracy
        is dominated by the majority class. This is why the evaluation reports <b>sensitivity, specificity, F1, PR-AUC</b> and event-level
        metrics rather than accuracy alone.</div></div>""", unsafe_allow_html=True)
    with c2:
        names = ["Baseline"] + [ui.MODEL_NAMES[k] for k in KEYS]
        fig = go.Figure()
        for metric, shade in [("accuracy", ui.INK), ("sensitivity", ui.ACCENT), ("f1", ui.ACCENT_SOFT)]:
            vals = [lazy[metric]] + [S[k]["window"][metric] for k in KEYS]
            fig.add_trace(go.Bar(x=names, y=vals, name=metric.capitalize() if metric != "f1" else "F1-score", marker=dict(color=shade),
                                 text=[f"{v * 100:.0f}%" for v in vals], textposition="outside", textfont=dict(family="IBM Plex Mono", size=11)))
        fig.update_layout(barmode="group", title="Accuracy versus sensitivity and F1", yaxis_range=[0, 1.15])
        fig = ui.style(fig, 400)
        fig.update_layout(legend=dict(y=-0.18, yanchor="top", x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True, theme=None)

with tabs[4]:
    st.markdown("Each detector trades **early detection** against **false alarms**. Adjust the decision rule and the "
                "event-level metrics are recomputed on the held-out hours.")
    c1, c2, c3 = st.columns(3)
    thr = c1.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
    k_of = c2.slider("Positive seconds required (K)", 1, 8, 3)
    n_of = c3.slider("Window length in seconds (N)", k_of, 10, max(4, k_of))
    out = {}
    for k in KEYS:
        recs = []
        for p in patients:
            if scope not in ("All patients", p):
                continue
            for r in R["patients"][p]["test_records"]:
                z = ui.prediction(r)
                recs.append({"record": r, "times": z["times"], "p": z[f"p_{k}"], "seizures": [tuple(s) for s in z["seizures"]], "duration": float(z["times"][-1] + 1)})
        out[k] = event_metrics(recs, thr, k_of, n_of)
    ui.cards([(ui.MODEL_NAMES[k],
               f"{out[k]['detected']}/{out[k]['seizures']}",
               f"latency {out[k]['mean_latency']:.1f} s · {out[k]['fa_per_hour']:.2f} false alarms / h" if out[k]['mean_latency'] is not None else f"{out[k]['fa_per_hour']:.2f} false alarms / h",
               ui.MODEL_COLORS[k]) for k in KEYS])

with tabs[5]:
    hist_patients = patients if scope == "All patients" else [scope]
    c1, c2 = st.columns(2)
    f1, f2 = go.Figure(), go.Figure()
    dashes = ["solid", "dash", "dot"]
    for j, p in enumerate(hist_patients):
        h = pd.DataFrame(R["patients"][p]["cnn_history"])
        f1.add_trace(go.Scatter(x=h.epoch, y=h.train_loss, name=f"{p} train", line=dict(color=ui.INK, dash=dashes[j], width=1.5)))
        f1.add_trace(go.Scatter(x=h.epoch, y=h.val_loss, name=f"{p} validation", line=dict(color=ui.MODEL_COLORS["cnn"], dash=dashes[j], width=1.5)))
        f2.add_trace(go.Scatter(x=h.epoch, y=h.val_acc, name=f"{p} val. accuracy", line=dict(color=ui.OK, dash=dashes[j], width=1.5)))
    f1.update_layout(title="Loss", xaxis_title="epoch", yaxis_title="cross-entropy loss")
    f2.update_layout(title="Validation accuracy", xaxis_title="epoch", yaxis_title="accuracy")
    f1 = ui.style(f1, 380)
    f1.update_layout(legend=dict(y=-0.22, yanchor="top", x=0.5, xanchor="center"))
    f2 = ui.style(f2, 380)
    f2.update_layout(legend=dict(y=-0.22, yanchor="top", x=0.5, xanchor="center"))
    c1.plotly_chart(f1, use_container_width=True, theme=None)
    c2.plotly_chart(f2, use_container_width=True, theme=None)
    st.caption("The 1D-CNN (about 110 k parameters) is trained with class-weighted binary cross-entropy, AdamW and a "
               "cosine schedule. Inputs are augmented with amplitude scaling, noise and time shifts; the checkpoint with "
               "the lowest validation loss is kept.")
