import datetime as dt
import html
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nsapp import ui
from neurosentinel.config import FS, WIN, CHANNELS, PATIENTS, MODEL_NAMES, MODEL_COLORS as REPORT_COLORS  # report stays a light document
from neurosentinel.data import available_records, edf_path, seizures_of, patient_of
from neurosentinel.edf import read_edf, select_channels
from neurosentinel.evaluate import alarm_signal
from neurosentinel.features import extract_features
from neurosentinel.preprocess import preprocess, window_starts, make_windows

ui.hero("Detect / Diagnose",
        "Screen a <em>recording</em>",
        "Upload an EDF file or choose one from the library. All three models scan it second by second, a consensus rule "
        "groups their alarms into events, and the result can be downloaded as a self-contained report.",
        index="06", meta="INPUT <b>EDF, 256 Hz</b><br>MONTAGE <b>18 bipolar channels</b><br>OUTPUT <b>HTML report</b>")
ui.need_artifacts()

REGIONS = {"left temporal": [0, 1, 2, 3], "left parasagittal": [4, 5, 6, 7], "right parasagittal": [8, 9, 10, 11],
           "right temporal": [12, 13, 14, 15], "midline": [16, 17]}

def _truth(n):
    """Annotated seizures if the file is a known CHB-MIT record, else None."""
    try:
        from neurosentinel.data import seizure_table
        tab = seizure_table(n.split("_")[0])
        return tab[n] if n in tab else None
    except Exception:
        return None


c1, c2 = st.columns([1.3, 1])
with c1:
    up = st.file_uploader("Upload an EDF recording", type=["edf"])
    lib = st.selectbox("Or choose from the library", ["—"] + available_records(), format_func=lambda r: "Select a recording" if r == "—" else ui.record_label(r))
with c2:
    name = up.name.rsplit(".", 1)[0] if up else (lib if lib != "—" else None)
    guess = name.split("_")[0] if name and name.split("_")[0] in PATIENTS else list(PATIENTS)[0]
    patient = st.selectbox("Patient-specific model set", list(PATIENTS), index=list(PATIENTS).index(guess),
                           help="Seizure detectors are patient-specific: each patient's seizures look different.")
    if name and name in [r for p in PATIENTS.values() for r in p["train"]] and patient == patient_of(name):
        st.info("This recording was part of the training set for that patient, so results will be optimistic.", icon=":material/info:")
    go_btn = st.button("Scan recording", icon=":material/play_arrow:", type="primary", use_container_width=True, disabled=name is None)

if go_btn and name:
    bar = st.progress(0, "Reading EDF…")
    t_start = time.perf_counter()
    rec = read_edf(up.getvalue() if up else edf_path(name))
    x = select_channels(rec, CHANNELS)
    bar.progress(15, "Filtering: 60 Hz notch + 0.5–40 Hz band-pass…")
    x = preprocess(x)
    starts = window_starts(x.shape[1], FS)
    times = (starts + WIN) / FS
    M = ui.models(patient)
    probs = {k: [] for k in MODEL_NAMES}
    ll = []
    for i in range(0, len(starts), 600):
        bar.progress(20 + int(75 * i / len(starts)), f"Classifying windows {i:,}–{min(len(starts), i + 600):,} of {len(starts):,}…")
        W = make_windows(x, starts[i:i + 600])
        F = extract_features(W)
        ll.append(np.abs(np.diff(W, axis=-1)).mean(-1))
        for k, v in M.predict_all(W, F).items():
            probs[k].append(v)
    probs = {k: np.concatenate(v) for k, v in probs.items()}
    ll = np.concatenate(ll)
    bar.progress(100, "Done.")
    st.session_state["dx"] = {"name": name, "patient": patient, "times": times, "probs": probs, "ll": ll,
                              "duration": x.shape[1] / FS, "elapsed": time.perf_counter() - t_start,
                              "truth": _truth(name) or [], "known": _truth(name) is not None}
    bar.empty()

dx = st.session_state.get("dx")
if not dx:
    ui.note("Suggested recordings: <b>chb01_21</b> and <b>chb03_35</b> are held-out hours containing a seizure; "
            "<b>chb01_06</b> is a seizure-free hour that shows the false-alarm behaviour.")
    st.stop()

times, probs, ll = dx["times"], dx["probs"], dx["ll"]
known = dx["known"]
alarms = {k: alarm_signal(v) for k, v in probs.items()}
consensus = (sum(a.astype(int) for a in alarms.values()) >= 2)
# merge consensus alarms into events
events, cur = [], None
for t, a in zip(times, consensus):
    if a and cur is None:
        cur = [t, t]
    elif a:
        cur[1] = t
    elif cur is not None and t - cur[1] > 10:
        events.append(cur); cur = None
if cur:
    events.append(cur)
base = np.median(ll, 0)
rows = []
for s, e in events:
    sel = (times >= s - 4) & (times <= e)
    act = (ll[sel] / base).mean(0)
    region = max(REGIONS, key=lambda r: act[REGIONS[r]].mean())
    truth = next(((a, b) for a, b in dx["truth"] if a - 30 <= s <= b + 30), None)
    rows.append({"Onset (alarm)": f"{int(s // 60):02d}:{int(s % 60):02d}", "Duration": f"{e - s + 4:.0f} s",
                 "Peak SVM": f"{probs['svm'][sel].max() * 100:.0f}%", "Peak RF": f"{probs['rf'][sel].max() * 100:.0f}%",
                 "Peak CNN": f"{probs['cnn'][sel].max() * 100:.0f}%", "Strongest region": region,
                 "Annotation": (f"matches seizure at {int(truth[0])} s" if truth else "no annotated seizure") if known else "unknown file"})

n_true = len(dx["truth"]) if known else "n/a"
ui.cards([
    ("Recording", dx["name"], f"{dx['duration'] / 60:.0f} min · scanned in {dx['elapsed']:.1f} s", ui.INK),
    ("Detected events", str(len(events)), "at least 2 of 3 models in alarm", ui.SEIZURE if events else ui.OK),
    ("Annotated seizures", str(n_true), "from the CHB-MIT summary file", ui.WARN),
    ("Peak probability", f"{max(v.max() for v in probs.values()) * 100:.0f}%", "highest single-window output", ui.ACCENT),
])

fig = go.Figure()
for a, b in dx["truth"]:
    fig.add_vrect(x0=a / 60, x1=b / 60, fillcolor=ui.SEIZURE, opacity=0.1, line_width=0, annotation_text="annotated", annotation_position="bottom left", annotation_font_color=ui.SEIZURE)
for s, e in events:
    fig.add_vrect(x0=s / 60, x1=(e + 4) / 60, line=dict(color=ui.INK, width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")
for k in MODEL_NAMES:
    fig.add_trace(go.Scatter(x=times / 60, y=probs[k], name=MODEL_NAMES[k], line=dict(color=ui.MODEL_COLORS[k], width=1.2)))
fig.add_trace(go.Scatter(x=times[consensus] / 60, y=np.full(consensus.sum(), 1.06), mode="markers", name="consensus alarm",
                         marker=dict(symbol="square", size=5, color=ui.SEIZURE)))
fig.update_layout(title="Model output per second (dotted boxes: detected events)")
fig.update_xaxes(title="minutes"); fig.update_yaxes(title="P(seizure)", range=[-0.02, 1.12])
fig = ui.style(fig, 380)
fig.update_layout(legend=dict(y=-0.22, yanchor="top", x=0.5, xanchor="center"))
st.plotly_chart(fig, use_container_width=True, theme=None)

if rows:
    st.markdown("#### Detected events")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.success("No seizure events detected in this recording.", icon=":material/check_circle:")


# ------------------------------------------------------------------ report
def svg_timeline(w=900, h=170):
    dur = dx["duration"]
    X = lambda t: 40 + t / dur * (w - 60)
    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#FFFFFF;border:1px solid #DCD9D1">']
    for a, b in dx["truth"]:
        parts.append(f'<rect x="{X(a):.1f}" y="10" width="{max(2, X(b) - X(a)):.1f}" height="{h - 40}" fill="#DA1E28" opacity="0.12"/>')
    for k in MODEL_NAMES:
        p = probs[k][::10]
        tt = times[::10]
        pts = " ".join(f"{X(t):.1f},{10 + (1 - v) * (h - 40):.1f}" for t, v in zip(tt, p))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{REPORT_COLORS[k]}" stroke-width="1.2"/>')
    for s, e in events:
        parts.append(f'<rect x="{X(s):.1f}" y="6" width="{max(3, X(e + 4) - X(s)):.1f}" height="{h - 32}" fill="none" stroke="#161616" stroke-dasharray="4 3"/>')
    for m in range(0, int(dur // 60) + 1, 10):
        parts.append(f'<text x="{X(m * 60):.1f}" y="{h - 8}" fill="#8D8D8D" font-size="10" font-family="IBM Plex Mono, Consolas, monospace" text-anchor="middle">{m}m</text>')
    parts.append("</svg>")
    return "".join(parts)


table = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>" for r in rows) or \
        "<tr><td colspan=7>No seizure events detected.</td></tr>"
head = "".join(f"<th>{h}</th>" for h in (rows[0].keys() if rows else ["Onset", "Duration", "Peak SVM", "Peak RF", "Peak CNN", "Region", "Annotation"]))
report = f"""<!doctype html><html><head><meta charset="utf-8"><title>NeuroSentinel report · {html.escape(dx['name'])}</title>
<style>body{{font-family:'IBM Plex Sans','Segoe UI',sans-serif;background:#F6F5F1;color:#161616;max-width:960px;margin:40px auto;padding:0 24px}}
h1{{font-family:Newsreader,Georgia,serif;font-weight:500;font-size:38px;margin:6px 0 4px}} .k{{font-family:'IBM Plex Mono',Consolas,monospace;color:#525252;letter-spacing:.14em;font-size:11px;text-transform:uppercase}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);margin:22px 0;border:1px solid #DCD9D1;background:#fff}} .c{{border-right:1px solid #DCD9D1;padding:14px;font-size:12.5px;color:#525252}} .c:last-child{{border-right:0}}
.c b{{display:block;font-size:24px;font-weight:500;color:#161616;font-family:'IBM Plex Mono',Consolas,monospace;margin-top:4px}} table{{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px;background:#fff}}
td,th{{border-bottom:1px solid #DCD9D1;padding:9px;text-align:left}} th{{color:#525252;font-weight:500;background:#ECEAE4}} small{{color:#525252}}
hr{{border:0;border-top:1px solid #161616;margin:14px 0 18px}}</style></head><body>
<div class="k">NeuroSentinel · automated EEG seizure screening</div><h1>Seizure report: {html.escape(dx['name'])}</h1><hr>
<small>Generated {dt.datetime.now():%Y-%m-%d %H:%M} · patient-specific models: {dx['patient']} · 18-channel bipolar montage · 4 s windows, 1 s stride</small>
<div class="grid"><div class="c">Duration<b>{dx['duration'] / 60:.0f} min</b></div><div class="c">Detected events<b>{len(events)}</b></div>
<div class="c">Annotated seizures<b>{n_true}</b></div><div class="c">Peak probability<b>{max(v.max() for v in probs.values()) * 100:.0f}%</b></div></div>
{svg_timeline()}
<p><small><span style="color:#6929C4">■ SVM</span> &nbsp; <span style="color:#005D5D">■ Random Forest</span> &nbsp; <span style="color:#1192E8">■ 1D-CNN</span> &nbsp;
<span style="color:#DA1E28">■ annotated seizure</span> &nbsp; ▭ detected event</small></p>
<table><tr>{head}</tr>{table}</table>
<p><small>Alarm rule: a model alarms when 3 of the last 4 one-second decisions exceed 50 %; an event is declared when at least two of the
three models alarm. This is a course research prototype, <b>not a medical device</b>.</small></p></body></html>"""
st.download_button("Download report (HTML)", report, icon=":material/download:", file_name=f"neurosentinel_{dx['name']}.html", mime="text/html",
                   use_container_width=False)
