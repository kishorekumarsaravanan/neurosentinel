import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from nsapp import ui
from neurosentinel.config import WIN, FS, BANDS
from neurosentinel.data import available_records, seizures_of
from neurosentinel.features import extract_features, PER_CHANNEL
from neurosentinel.preprocess import window_starts, make_windows

ui.hero("Discover / Brain Atlas",
        "One hour of EEG, <em>at a glance</em>",
        "The activity map shows every channel minute by minute across a full recording, and seizures stand out as "
        "bands of high activity. The scalp time-lapse below shows where a seizure <b>starts and how it spreads</b>.",
        index="02", meta="WINDOW <b>4 s</b><br>FEATURE <b>line length, band power</b><br>SCALE <b>YlOrRd / inferno</b>")

records = available_records()
if not records:
    st.error("No EDF files in data/raw yet. Run `bash scripts/download_data.sh`.")
    st.stop()
sz_recs = [r for r in records if seizures_of(r)]
default = next((r for r in ui.test_records() if r in sz_recs), sz_recs[0] if sz_recs else records[0])
rec = st.selectbox("Recording", records, index=records.index(default), format_func=ui.record_label)


@st.cache_data(show_spinner="Computing channel activity…", max_entries=4)
def overview(rec: str):
    x = ui.clean_signal(rec)
    st_ = window_starts(x.shape[1], WIN)
    F = extract_features(make_windows(x, st_)).reshape(len(st_), len(ui.CHANNELS), len(PER_CHANNEL))
    t = (st_ + WIN / 2) / FS
    i0 = PER_CHANNEL.index("rel_power_delta")
    return {"t": t, "ll": F[..., PER_CHANNEL.index("line_length")], "rel": F[..., i0:i0 + 5],
            "logpow": F[..., PER_CHANNEL.index("log_power_delta"):PER_CHANNEL.index("log_power_delta") + 5]}


ov = overview(rec)
sz = seizures_of(rec)
t_min = ov["t"] / 60
ll = ov["ll"]
llz = (ll - np.median(ll, 0)) / (ll.std(0) + 1e-6)

has_pred = rec in ui.test_records() and ui.artifacts_ready()
rows = 3 if has_pred else 2
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                    row_heights=[0.55, 0.25, 0.2][:rows] if has_pred else [0.68, 0.32])
fig.add_trace(go.Heatmap(z=llz.T, x=t_min, y=ui.CHANNELS, zmin=0, zmax=6,
                         colorscale=ui.scale(ui.ACTIVITY),
                         showscale=False, hovertemplate="%{y}<br>%{x:.1f} min<br>activity z=%{z:.1f}<extra></extra>"), 1, 1)
band_cols = ui.BANDS_COL
rel = ov["rel"].mean(1)
for j, b in enumerate(BANDS):
    fig.add_trace(go.Scatter(x=t_min, y=rel[:, j], stackgroup="b", name=b, line=dict(width=0.3, color=band_cols[j]), fillcolor=band_cols[j]), 2, 1)
if has_pred:
    z = ui.prediction(rec)
    for k in ["svm", "rf", "cnn"]:
        fig.add_trace(go.Scatter(x=z["times"] / 60, y=z[f"p_{k}"], name=ui.MODEL_NAMES[k], line=dict(color=ui.MODEL_COLORS[k], width=1)), 3, 1)
for s, e in sz:
    for r in range(1, rows + 1):
        fig.add_vrect(x0=s / 60, x1=e / 60, line=dict(color=ui.SEIZURE, width=1), fillcolor="rgba(0,0,0,0)", row=r, col=1)
fig.update_yaxes(tickfont=dict(family="IBM Plex Mono", size=9), row=1, col=1, autorange="reversed")
fig.update_yaxes(title="band mix", range=[0, 1], row=2, col=1)
if has_pred:
    fig.update_yaxes(title="P(seizure)", range=[0, 1], row=3, col=1)
fig.update_xaxes(title="minutes", row=rows, col=1)
fig.update_layout(title="Channel activity (line length, z-scored), frequency mix" + (" and model output" if has_pred else ""))
fig = ui.style(fig, 660)
fig.update_layout(legend=dict(y=-0.08, yanchor="top", x=0, xanchor="left"))
st.plotly_chart(fig, width="stretch", theme=None)

# ------------------------------------------------------------- time-lapse
st.markdown("#### Scalp time-lapse")
c1, c2 = st.columns([1, 2])
metric = c1.selectbox("Colour the scalp by", ["Activity (line length)"] + [f"{b.capitalize()} power ({lo:g}–{hi:g} Hz)" for b, (lo, hi) in BANDS.items()])
dur_min = float(ov["t"][-1] / 60)
center = (sz[0][0] + sz[0][1]) / 2 / 60 if sz else dur_min / 2
span = c2.slider("Time range (minutes)", 0.0, dur_min, (max(0.0, center - 1.6), min(dur_min, center + 1.6)), 0.1)
sel = (t_min >= span[0]) & (t_min <= span[1])
if metric.startswith("Activity"):
    vals = ll
else:
    vals = ov["logpow"][..., list(BANDS).index(metric.split()[0].lower())]
idx = np.flatnonzero(sel)
# 0 = this channel's typical level over the hour, 1 = the strongest deviation inside the selected span,
# so relative differences between regions (where the storm starts / spreads) stay visible
lo = np.median(vals, axis=0)
hi = np.percentile((vals[idx] - lo), 99) if len(idx) else 1.0
norm = np.clip((vals - lo) / (hi + 1e-9), 0, 1) * 0.92 + 0.04
if len(idx) > 240:
    idx = idx[:: int(np.ceil(len(idx) / 240))]
frames = norm[idx].round(3).tolist()
labels = [f"t = {int(t // 60):02d}:{int(t % 60):02d}" for t in ov["t"][idx]]
marks = [any(s <= t <= e for s, e in sz) for t in ov["t"][idx]]
start = next((i for i, m in enumerate(marks) if m), 0)
ui.component("brainmap", {"channels": ui.CHANNELS, "frames": frames, "labels": labels, "marks": marks,
                          "raw": vals[idx].round(2).tolist(), "fps": 3, "start": max(0, start - 6),
                          "title": f"{rec} · {metric} · one frame per 4 s · red = annotated seizure",
                          "list_title": "Most active channels"}, height=520)

# ------------------------------------------------------------- EEG viewer
st.markdown("#### EEG page view")
x = ui.clean_signal(rec)
dur = x.shape[1] / FS
c1, c2 = st.columns([3, 1])
t0 = c1.slider("Start (s)", 0.0, float(dur - 20), float(max(0.0, sz[0][0] - 8)) if sz else 60.0, 1.0)
gain = c2.select_slider("Amplitude", [0.5, 1, 2, 4], value=1)
seg = x[:, int(t0 * FS):int((t0 + 20) * FS):2]
tt = t0 + np.arange(seg.shape[1]) * 2 / FS
fig = go.Figure()
for s, e in sz:
    if e > t0 and s < t0 + 20:
        fig.add_vrect(x0=max(s, t0), x1=min(e, t0 + 20), fillcolor=ui.SEIZURE, opacity=0.08, line_width=0)
for i, ch in enumerate(ui.CHANNELS):
    fig.add_trace(go.Scatter(x=tt, y=i - seg[i] / 250 * gain, mode="lines", line=dict(width=0.8, color=ui.INK), name=ch, showlegend=False,
                             hovertemplate=f"{ch}<br>%{{x:.2f}} s<extra></extra>"))
fig.update_yaxes(tickvals=list(range(len(ui.CHANNELS))), ticktext=ui.CHANNELS, autorange="reversed", range=[len(ui.CHANNELS), -1],
                 tickfont=dict(family="IBM Plex Mono", size=10, color=ui.INK), showgrid=False)
fig.update_xaxes(title="seconds", dtick=1)
fig.update_layout(title="20-second page, 0.5–40 Hz band-pass (red: annotated seizure)")
st.plotly_chart(ui.style(fig, 620, legend=False), width="stretch", theme=None)
