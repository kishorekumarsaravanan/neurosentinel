import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nsapp import ui
from neurosentinel.config import MODELS_DIR, WIN, FS
from neurosentinel.data import seizures_of, patient_of
from neurosentinel.features import extract_features, FEATURE_NAMES, PER_CHANNEL, FEATURE_FAMILY

ui.hero("Evaluate / Explainability",
        "Why did the model <em>raise the alarm?</em>",
        "Clinicians need to see the evidence behind a decision. Choose any moment of a held-out recording: each channel "
        "is masked in turn to measure how much it contributed to every model's verdict, and the CNN's gradients are "
        "traced back to the samples it relied on.",
        index="08", meta="METHODS <b>occlusion</b>, <b>gradient × input</b><br>GLOBAL <b>random-forest importance</b>")
ui.need_artifacts()

recs = [r for r in ui.test_records() if seizures_of(r)]
c1, c2 = st.columns([1, 2], gap="large")
rec = c1.selectbox("Held-out recording", recs, format_func=ui.record_label)
s0, e0 = seizures_of(rec)[0]
x = ui.clean_signal(rec)
dur = x.shape[1] / FS
preset = c1.radio("Moment", ["Seizure onset + 8 s", "Mid-seizure", "60 s before onset", "Background"], horizontal=False)
default = {"Seizure onset + 8 s": s0 + 8, "Mid-seizure": (s0 + e0) / 2, "60 s before onset": s0 - 60,
           "Background": max(30.0, s0 - 600)}[preset]
t_end = c2.slider("Window end time (s)", 4.0, float(dur), float(min(dur, default)), 1.0)
start = int((t_end - 4) * FS)
W = x[:, start:start + WIN][None].astype(np.float32)
F = extract_features(W)
M = ui.models(patient_of(rec))
probs = M.predict_all(W, F)
in_sz = any(s <= t_end - 2 <= e for s, e in seizures_of(rec))

with c2:
    ui.cards([(ui.MODEL_NAMES[k], f"{probs[k][0] * 100:.0f}%", "P(seizure)", ui.MODEL_COLORS[k]) for k in ["svm", "rf", "cnn"]]
             + [("Annotation", "Ictal" if in_sz else "Non-ictal", "neurologist label", ui.SEIZURE if in_sz else ui.OK)])


# ----------------------------------------------------------- occlusion: mask one channel at a time
@st.cache_data(show_spinner=False)
def occlusion(rec: str, t_end: float):
    bg = np.load(MODELS_DIR / patient_of(rec) / "background_feature_median.npy")
    nf = len(PER_CHANNEL)
    Fo = np.repeat(F, len(ui.CHANNELS), 0)
    Wo = np.repeat(W, len(ui.CHANNELS), 0)
    for c in range(len(ui.CHANNELS)):
        Fo[c, c * nf:(c + 1) * nf] = bg[c * nf:(c + 1) * nf]
        Wo[c, c] = 0.0
    po = M.predict_all(Wo, Fo)
    return {k: np.clip(probs[k][0] - po[k], 0, None) for k in po}


occ = occlusion(rec, t_end)
st.markdown("#### Channel contribution")
st.caption("Occlusion analysis: each channel is replaced by its typical background behaviour and the drop in seizure "
           "probability is recorded. Stronger colour means that region pushed the model further towards a seizure decision.")
cols = st.columns(3, gap="medium")
for col, k in zip(cols, ["svm", "rf", "cnn"]):
    v = occ[k]
    norm = (v / v.max()).tolist() if v.max() > 1e-6 else [0.0] * len(v)
    with col:
        st.markdown(f"<div style='font-weight:600;font-size:14.5px;margin-bottom:6px'>{ui.model_label(k, 16)}"
                    f"<span style='color:var(--ink3);font-weight:400;font-family:IBM Plex Mono;font-size:12px'> &nbsp;top channel {ui.CHANNELS[int(np.argmax(v))]}</span></div>",
                    unsafe_allow_html=True)
        ui.component("brainmap", {"channels": ui.CHANNELS, "frames": [norm], "stops": "xray", "compact": True,
                                  "title": "", "lo": "no effect", "hi": "decisive"}, height=330)

# ----------------------------------------------------------- CNN saliency
st.markdown("#### CNN attention over the window")
sal = M.cnn.saliency(W[0])
k = 8
sal_ds = sal.reshape(sal.shape[0], -1, k).mean(-1)
sal_ds = sal_ds / (sal_ds.max() + 1e-9)
tt = np.arange(sal_ds.shape[1]) * k / FS + (t_end - 4)
fig = go.Figure(go.Heatmap(z=sal_ds, x=tt, y=list(range(len(ui.CHANNELS))), customdata=[[c] * sal_ds.shape[1] for c in ui.CHANNELS],
                           colorscale=ui.scale(ui.IMPORTANCE), showscale=False,
                           hovertemplate="%{customdata} @ %{x:.2f}s<br>saliency %{z:.2f}<extra></extra>"))
ts = np.arange(WIN) / FS + (t_end - 4)
for i, ch in enumerate(ui.CHANNELS):
    sig = W[0, i] / (np.percentile(np.abs(W[0]), 99) + 1e-6) * 0.45
    fig.add_trace(go.Scatter(x=ts[::2], y=i - sig[::2], mode="lines", line=dict(color=ui.INK, width=0.9), showlegend=False, hoverinfo="skip"))
fig.update_yaxes(tickvals=list(range(len(ui.CHANNELS))), ticktext=ui.CHANNELS, range=[len(ui.CHANNELS) - 0.4, -0.6], showgrid=False)
fig.update_xaxes(title="time in recording (s)", showgrid=False)
fig.update_layout(title="Gradient × input saliency (stronger colour = more influential samples)")
st.plotly_chart(ui.style(fig, 560, legend=False), use_container_width=True, theme=None)

# ----------------------------------------------------------- global feature importance
st.markdown("#### What the random forest learned")
imp = np.array(ui.results()["patients"][patient_of(rec)]["rf_importance"])
df = pd.DataFrame({"name": FEATURE_NAMES, "imp": imp})
df["channel"] = df.name.str.split("|").str[0]
df["feature"] = df.name.str.split("|").str[1]
df["family"] = df.feature.map(FEATURE_FAMILY).map({"time": "Time domain", "frequency": "Frequency domain"})
c1, c2 = st.columns([1.1, 1], gap="medium")
with c1:
    top = df.groupby(["family", "feature"], as_index=False).imp.sum()
    fig = go.Figure(go.Sunburst(
        labels=["All features", "Time domain", "Frequency domain"] + top.feature.str.replace("_", " ").tolist(),
        parents=["", "All features", "All features"] + top.family.tolist(),
        values=[0, 0, 0] + top.imp.tolist(),
        branchvalues="remainder",
        marker=dict(colors=[ui.INSET, ui.INK, ui.ACCENT] + [ui.FILL_MUTED2 if f == "Time domain" else ui.ACCENT_SOFT for f in top.family],
                    line=dict(color=ui.SURFACE, width=1.5)),
        textfont=dict(family="IBM Plex Sans", color=ui.PAPER),
        hovertemplate="%{label}<br>%{value:.3f}<extra></extra>", insidetextorientation="radial"))
    fig.update_layout(title="Importance by feature family (click to zoom)")
    st.plotly_chart(ui.style(fig, 470, legend=False), use_container_width=True, theme=None)
with c2:
    t15 = df.sort_values("imp", ascending=False).head(15)[::-1]
    fig = go.Figure(go.Bar(x=t15.imp, y=t15.channel + " · " + t15.feature.str.replace("_", " "), orientation="h",
                           marker=dict(color=ui.INK)))
    fig.update_layout(title="Top 15 individual features")
    fig.update_yaxes(tickfont=dict(family="IBM Plex Mono", size=10, color=ui.INK), showgrid=False)
    st.plotly_chart(ui.style(fig, 470, legend=False), use_container_width=True, theme=None)
