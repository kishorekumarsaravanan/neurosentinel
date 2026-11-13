"""NeuroSentinel: EEG seizure detection workbench.

Run:  streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="NeuroSentinel", page_icon="nsapp/assets/mark.svg", layout="wide",
                   initial_sidebar_state="expanded")

from nsapp import ui  # noqa: E402

ui.setup()

V = "nsapp/views/"
pages = {
    "Discover": [
        st.Page(V + "prologue.py", title="Overview", icon=":material/auto_stories:", default=True),
        st.Page(V + "atlas.py", title="Brain Atlas", icon=":material/explore:"),
        st.Page(V + "signal_lab.py", title="Signal Lab", icon=":material/tune:"),
    ],
    "Detect": [
        st.Page(V + "monitor.py", title="Live Monitor", icon=":material/monitor_heart:"),
        st.Page(V + "game.py", title="Seizure Hunter", icon=":material/sports_esports:"),
        st.Page(V + "diagnose.py", title="Diagnose", icon=":material/stethoscope:"),
    ],
    "Evaluate": [
        st.Page(V + "arena.py", title="Model Comparison", icon=":material/leaderboard:"),
        st.Page(V + "xray.py", title="Explainability", icon=":material/radiology:"),
    ],
}

st.logo("nsapp/assets/logo.svg", size="large", icon_image="nsapp/assets/mark.svg")
nav = st.navigation(pages)

with st.sidebar:
    ui.theme_toggle()
    ready = ui.artifacts_ready()
    status = ('<span style="color:#42BE65">●</span> models loaded' if ready
              else '<span style="color:#F1C21B">●</span> run train.py')
    st.markdown(
        f"""<div style="font-family:'IBM Plex Mono';font-size:11.5px;line-height:1.9;color:#A8A8A8;border-top:1px solid #393939;padding-top:14px;margin-top:6px">
        <div>DATA&nbsp;&nbsp;&nbsp;<span style="color:#F4F4F4">CHB-MIT scalp EEG</span></div>
        <div>MODELS&nbsp;<span style="color:#F4F4F4">SVM · RF · 1D-CNN</span></div>
        <div>STATUS&nbsp;{status}</div>
        <div>BUILD&nbsp;&nbsp;<span style="color:#F4F4F4">v1.0 · BCI course project</span></div>
        </div>""", unsafe_allow_html=True)

nav.run()
