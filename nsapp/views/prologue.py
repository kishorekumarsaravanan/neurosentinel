import numpy as np
import streamlit as st
from nsapp import ui
from neurosentinel.config import FS
from neurosentinel.data import available_records, seizures_of

ui.hero("Discover / Overview",
        "Detecting epileptic seizures <em>within seconds</em>, from scalp EEG.",
        "NeuroSentinel compares three detectors (a support-vector machine, a random forest and a 1D convolutional "
        "network) on the CHB-MIT paediatric EEG database. Every model is evaluated on recordings it has never seen, "
        "with the metrics a neurologist would ask for: sensitivity, false alarms per hour and detection latency.",
        index="01", meta="DATASET <b>CHB-MIT</b><br>PATIENTS <b>chb01, chb03</b><br>RECORDINGS <b>21 h</b><br>VALIDATION <b>patient-specific</b>")


@st.cache_data(show_spinner=False)
def dot_matrix(rec: str = "chb01_21", cols: int = 160, ink: str = "#161616", red: str = "#DA1E28") -> str:   # 15360 samples / 160 = 96 per column
    """One minute of 18-channel activity drawn as a dot matrix; red columns = annotated seizure."""
    try:
        x = ui.clean_signal(rec)
        s, e = seizures_of(rec)[0]
    except Exception:
        return ""
    t0 = s - 32
    seg = x[:, int(t0 * FS):int((t0 + 60) * FS)]
    env = np.abs(seg).reshape(seg.shape[0], cols, -1).mean(-1)
    env = env / np.percentile(env, 97)
    w, h, gx, gy = 1200, 190, 1200 / cols, 190 / 18
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">']
    for c in range(cols):
        ictal = s <= t0 + (c + 0.5) * 60 / cols <= e
        col = red if ictal else ink
        for r in range(18):
            rad = min(gx, gy) * 0.46 * min(1.0, 0.18 + 0.82 * env[r, c])
            out.append(f'<circle cx="{(c + 0.5) * gx:.1f}" cy="{(r + 0.5) * gy:.1f}" r="{rad:.2f}" fill="{col}"/>')
    out.append("</svg>")
    return "".join(out)


svg = dot_matrix(ink=ui.INK, red=ui.SEIZURE) if "chb01_21" in available_records() else ""
if svg:
    st.markdown(f'<div style="background:var(--surface);border:1px solid var(--line);padding:18px 18px 10px">{svg}'
                f'<div style="display:flex;justify-content:space-between;font-family:IBM Plex Mono;font-size:11px;color:var(--ink3);margin-top:8px">'
                f'<span>FIG. 1 · 60 s of 18-channel scalp EEG, record chb01_21. Dot size = local signal amplitude.</span>'
                f'<span><span style="color:var(--seizure)">●</span> annotated seizure &nbsp; ● background</span></div></div>',
                unsafe_allow_html=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
c1, c2 = st.columns([1, 2], gap="large")
with c1:
    st.markdown("### What is a seizure?")
with c2:
    st.markdown("<div style='color:var(--ink2);font-size:15.5px;line-height:1.65;padding-top:6px'>Cortical neurons normally fire on "
                "their own schedules, and their electric fields largely cancel at the scalp. During a seizure they lock into "
                "a shared rhythm (<b style='color:var(--ink)'>hypersynchrony</b>), and the fields add up into the large, regular "
                "waves that the detectors learn to recognise. The simulation below models 160 neurons as coupled Kuramoto "
                "oscillators: raise the coupling past its critical value (about 3.5) and the population synchronises.</div>",
                unsafe_allow_html=True)
ui.component("hypersync", {}, height=470)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
ui.cards([
    ("People living with epilepsy", "50 M", "worldwide, WHO estimate", ui.INK),
    ("Ictal share of recordings", "< 1 %", "seizures are rare events", ui.SEIZURE),
    ("Sampling rate", "256 Hz", "18 bipolar channels used", ui.ACCENT),
    ("Detectors compared", "3", "SVM, Random Forest, 1D-CNN", ui.MODEL_COLORS["svm"]),
])

st.markdown("### Method")
steps = [
    ("01", "Acquire", "Scalp EEG from the CHB-MIT database (Boston Children's Hospital): 18 bipolar channels of the longitudinal montage at 256 Hz."),
    ("02", "Filter", "A 60 Hz notch removes mains interference; a zero-phase 0.5–40 Hz Butterworth band-pass removes drift and muscle activity."),
    ("03", "Segment", "The signal is cut into 4-second windows. Seizure windows overlap by 75 % during training to augment the rare class."),
    ("04", "Extract", "21 time- and frequency-domain features per channel (band powers, line length, Hjorth parameters, spectral entropy and others), 378 in total."),
    ("05", "Classify", "The SVM and the random forest use the features; the 1D-CNN learns its own representation from the filtered signal."),
    ("06", "Alarm", "An alarm is raised when 3 of the last 4 one-second decisions are positive. Results are reported per window and per seizure event."),
]
c1, c2 = st.columns([1.5, 1], gap="large")
with c1:
    ui.panel("".join(f'<div class="ns-step"><div class="n">{n}</div><div><b>{t}</b><div>{d}</div></div></div>' for n, t, d in steps))
with c2:
    st.markdown('<div class="ns-label" style="margin-bottom:8px">Explore</div>', unsafe_allow_html=True)
    for page, label, ic in [("monitor.py", "Live Monitor: replay a held-out seizure", ":material/monitor_heart:"),
                            ("game.py", "Seizure Hunter: compete with the models", ":material/sports_esports:"),
                            ("arena.py", "Model Comparison: metrics and trade-offs", ":material/leaderboard:"),
                            ("xray.py", "Explainability: what drove each decision", ":material/radiology:"),
                            ("atlas.py", "Brain Atlas: one hour of EEG at a glance", ":material/explore:"),
                            ("signal_lab.py", "Signal Lab: preprocessing, step by step", ":material/tune:"),
                            ("diagnose.py", "Diagnose: scan your own EDF file", ":material/stethoscope:")]:
        st.page_link("nsapp/views/" + page, label=label, icon=ic)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ui.note("Validation is <b>patient-specific</b>: models are tested only on recordings excluded from training, "
            "including complete seizure-free hours, so the false-alarm rate is measured honestly.")
