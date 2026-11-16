import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import signal as sps
from nsapp import ui
from neurosentinel.config import FS, WIN, BANDS
from neurosentinel.data import available_records, seizures_of
from neurosentinel.preprocess import bandpass, notch
from neurosentinel.features import extract_features, PER_CHANNEL

ui.hero("Discover / Signal Lab",
        "From raw signal to <em>features</em>",
        "Raw EEG contains mains interference, slow drift and muscle noise. Adjust the filter settings to see the effect "
        "of each preprocessing step, then follow one 4-second window as it becomes a vector of 21 features per channel.",
        index="03", meta="NOTCH <b>60 Hz</b><br>BAND-PASS <b>0.5–40 Hz</b><br>FEATURES <b>21 × 18 = 378</b>")

records = [r for r in available_records() if seizures_of(r)] or available_records()
if not records:
    st.error("No EDF files in data/raw yet. Run `bash scripts/download_data.sh`.")
    st.stop()
c1, c2, c3 = st.columns([1.4, 1, 1])
rec = c1.selectbox("Recording", records, format_func=ui.record_label)
ch = c2.selectbox("Channel", ui.CHANNELS, index=ui.CHANNELS.index("F7-T7"))
sz = seizures_of(rec)
raw = ui.raw_signal(rec)
ci = ui.CHANNELS.index(ch)
dur = raw.shape[1] / FS
t_c = c3.number_input("Focus time (s)", 0.0, dur - 10, float(sz[0][0] + 5) if sz else 120.0, 5.0)

st.markdown("#### 1 · Filtering")
k1, k2, k3, k4 = st.columns(4)
lo = k1.slider("High-pass (Hz)", 0.1, 5.0, 0.5, 0.1)
hi = k2.slider("Low-pass (Hz)", 15.0, 100.0, 40.0, 1.0)
order = k3.select_slider("Butterworth order", [2, 4, 6, 8], value=4)
use_notch = k4.toggle("60 Hz notch", True)

seg0 = int(max(0, t_c - 30) * FS)
seg1 = int(min(dur, t_c + 30) * FS)
x_raw = raw[ci, seg0:seg1].astype(np.float64)
x_f = notch(x_raw, 60.0) if use_notch else x_raw
x_f = bandpass(x_f, lo, hi, FS, order)
tt = seg0 / FS + np.arange(len(x_raw)) / FS
view = (tt >= t_c - 5) & (tt <= t_c + 5)

fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], subplot_titles=["10 s of signal", "Power spectrum (Welch)"])
fig.add_trace(go.Scatter(x=tt[view], y=x_raw[view], name="raw", line=dict(color=ui.FILL_MUTED, width=1)), 1, 1)
fig.add_trace(go.Scatter(x=tt[view], y=x_f[view], name="filtered", line=dict(color=ui.INK, width=1.1)), 1, 1)
f, p_raw = sps.welch(x_raw, FS, nperseg=FS * 2)
_, p_f = sps.welch(x_f, FS, nperseg=FS * 2)
band_cols = ui.BANDS_COL
for (b, (bl, bh)), c in zip(BANDS.items(), band_cols):
    fig.add_vrect(x0=bl, x1=bh, fillcolor=c, opacity=0.08, line_width=0, row=1, col=2, annotation_text=b, annotation_position="top left",
                  annotation_font=dict(size=10, color=c))
fig.add_trace(go.Scatter(x=f, y=10 * np.log10(p_raw + 1e-12), name="raw PSD", line=dict(color=ui.FILL_MUTED)), 1, 2)
fig.add_trace(go.Scatter(x=f, y=10 * np.log10(p_f + 1e-12), name="filtered PSD", line=dict(color=ui.ACCENT, width=1.8)), 1, 2)
fig.update_xaxes(title="s", row=1, col=1)
fig.update_xaxes(title="Hz", range=[0, 128], row=1, col=2)
fig.update_yaxes(title="µV", row=1, col=1)
fig.update_yaxes(title="dB", row=1, col=2)
fig = ui.style(fig, 380)
fig.update_layout(legend=dict(y=-0.28, yanchor="top", x=0.5, xanchor="center"))
st.plotly_chart(fig, width="stretch", theme=None)

st.markdown("#### 2 · Time–frequency view")
f_s, t_s, S = sps.spectrogram(x_f, FS, nperseg=FS, noverlap=int(FS * 0.75))
keep = f_s <= 45
fig = go.Figure(go.Heatmap(z=10 * np.log10(S[keep] + 1e-9), x=t_s + seg0 / FS, y=f_s[keep], showscale=False,
                           colorscale=ui.scale(ui.SPECTRO)))
for s, e in sz:
    fig.add_vrect(x0=s, x1=e, line=dict(color=ui.SEIZURE, width=2, dash="dot"), fillcolor="rgba(0,0,0,0)")
fig.update_layout(title=f"Spectrogram of {ch}, 60 s around the focus (dotted box: annotated seizure)")
fig.update_yaxes(title="Hz")
fig.update_xaxes(title="s", range=[seg0 / FS, seg1 / FS])
st.plotly_chart(ui.style(fig, 300, legend=False), width="stretch", theme=None)

st.markdown("#### 3 · Segmentation")
st.caption("Background windows are taken without overlap; seizure windows slide by 1 s (75 % overlap) during training to "
           "multiply the rare class. A window is labelled 'seizure' when at least half of it lies inside the annotation.")
fig = go.Figure()
fig.add_trace(go.Scatter(x=tt, y=x_f, line=dict(color=ui.INK2, width=0.7), showlegend=False))
amp = np.percentile(np.abs(x_f), 99)
for s in np.arange(seg0 / FS, seg1 / FS - 4, 4):
    frac = sum(max(0, min(e, s + 4) - max(a, s)) for a, e in sz) / 4
    col = ui.SEIZURE if frac >= 0.5 else ui.FILL_MUTED2
    fig.add_shape(type="rect", x0=s + 0.05, x1=s + 3.95, y0=-amp * 1.25, y1=-amp * 1.1, fillcolor=col, opacity=0.8, line_width=0)
fig.update_xaxes(title="s"); fig.update_yaxes(visible=False)
st.plotly_chart(ui.style(fig, 230, legend=False), width="stretch", theme=None)

st.markdown("#### 4 · Feature vector of one window")
clean = ui.clean_signal(rec)
w0 = int((t_c - 2) * FS)
Wf = clean[:, w0:w0 + WIN][None]
F = extract_features(Wf).reshape(len(ui.CHANNELS), len(PER_CHANNEL))
bg_t = max(30.0, (sz[0][0] - 400) if sz else 60.0)
B = extract_features(np.stack([clean[:, int((bg_t + 4 * i) * FS):int((bg_t + 4 * i) * FS) + WIN] for i in range(40)])).reshape(40, len(ui.CHANNELS), len(PER_CHANNEL))
z = (F[ci] - B[:, ci].mean(0)) / (B[:, ci].std(0) + 1e-6)
zc = np.clip(z, -6, 6)
nice = [n.replace("_", " ") for n in PER_CHANNEL]
c1, c2 = st.columns([1, 1])
with c1:
    fig = go.Figure(go.Barpolar(r=np.abs(zc), theta=nice, marker=dict(color=[ui.SEIZURE if v > 2 else ui.FILL_MUTED2 if v > -2 else ui.ACCENT for v in zc],
                                                                         line=dict(color=ui.SURFACE, width=1)), opacity=0.95,
                                hovertemplate="%{theta}<br>|z| = %{r:.1f}<extra></extra>"))
    fig.update_layout(title=f"Deviation from background, {ch} (|z|)",
                      polar=dict(radialaxis=dict(range=[0, 6], gridcolor=ui.GRID_STRONG, tickfont=dict(size=8, family="IBM Plex Mono")),
                                 angularaxis=dict(tickfont=dict(size=9.5, color=ui.INK2), gridcolor=ui.GRID, linecolor=ui.LINE_STRONG)))
    st.plotly_chart(ui.style(fig, 470, legend=False, margin=dict(l=70, r=70, t=64, b=30)), width="stretch", theme=None)
with c2:
    heat = (F - B.mean(0)) / (B.std(0) + 1e-6)
    fig = go.Figure(go.Heatmap(z=np.clip(heat, -6, 6), x=nice, y=ui.CHANNELS, zmid=0, colorscale=ui.scale(ui.DIVERGING),
                               showscale=False, hovertemplate="%{y} · %{x}<br>z = %{z:.1f}<extra></extra>"))
    fig.update_layout(title="All 18 channels × 21 features (z-scores against background)")
    fig.update_yaxes(autorange="reversed", tickfont=dict(family="IBM Plex Mono", size=9))
    fig.update_xaxes(tickfont=dict(size=9), tickangle=-50)
    st.plotly_chart(ui.style(fig, 470, legend=False), width="stretch", theme=None)
st.caption("Red: far above this patient's normal background; blue: far below. During a seizure, line length, variance "
           "and low-frequency power rise sharply while spectral entropy falls, because the signal becomes rhythmic and predictable.")
