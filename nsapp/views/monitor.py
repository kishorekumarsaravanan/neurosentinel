import numpy as np
import plotly.graph_objects as go
import streamlit as st
from nsapp import ui
from neurosentinel.evaluate import alarm_signal

ui.hero("Detect / Live Monitor",
        "Live seizure <em>monitor</em>",
        "Replay of EEG the models were never trained on. All three detectors judge every second, and an alarm is raised "
        "when two of them agree. The scalp map tracks where activity is rising, the assessment line summarises the "
        "current state, and <b>Sound</b> turns the signal envelope into audio. Space bar plays and pauses.",
        index="04", meta="WINDOW <b>4 s</b><br>STRIDE <b>1 s</b><br>ALARM RULE <b>3 of 4 s</b><br>CONSENSUS <b>2 of 3 models</b>")
ui.need_artifacts()

clips = sorted(ui.clips(), key=lambda c: (c["kind"] != "seizure", c["record"]))
labels = [f"{c['title']}  ·  {'contains a seizure' if c['kind'] == 'seizure' else 'seizure-free control'}" for c in clips]
c1, c2 = st.columns([3, 1])
with c1:
    choice = st.selectbox("Recording", range(len(clips)), format_func=lambda i: labels[i])
with c2:
    speed = st.select_slider("Initial speed", [1, 2, 5, 10], value=2, format_func=lambda v: f"{v}×")
clip = clips[choice]
ui.component("monitor", {"clip": clip, "speed": speed, "autoplay": False}, height=860)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.markdown("#### The full hour, in retrospect")
st.caption("Seizure probability from each model across the complete held-out recording. Red bands are the neurologist's "
           "annotations; the ticks below each trace mark that model's alarms. The dotted box is the replayed clip.")
pred = ui.prediction(clip["record"])
t = pred["times"] / 60
fig = go.Figure()
for s, e in pred["seizures"]:
    fig.add_vrect(x0=s / 60, x1=e / 60, fillcolor=ui.SEIZURE, opacity=0.12, line_width=0)
fig.add_vrect(x0=clip["t0"] / 60, x1=(clip["t0"] + clip["duration"]) / 60, fillcolor="rgba(0,0,0,0)",
              line=dict(color=ui.INK2, width=1, dash="dot"))
for j, k in enumerate(["svm", "rf", "cnn"]):
    p = pred[f"p_{k}"]
    fig.add_trace(go.Scatter(x=t, y=p, name=ui.MODEL_NAMES[k], line=dict(color=ui.MODEL_COLORS[k], width=1.2)))
    al = alarm_signal(p)
    fig.add_trace(go.Scatter(x=t[al], y=np.full(al.sum(), -0.08 - 0.07 * j), mode="markers", showlegend=False,
                             marker=dict(symbol="line-ns", size=8, line=dict(width=2, color=ui.MODEL_COLORS[k])),
                             hoverinfo="skip"))
fig.add_hline(y=0.5, line=dict(color=ui.WARN, dash="dash", width=1))
fig.update_yaxes(range=[-0.32, 1.05], title="P(seizure)")
fig.update_xaxes(title="minutes into recording")
st.plotly_chart(ui.style(fig, 340), use_container_width=True, theme=None)
