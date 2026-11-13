import streamlit as st
from nsapp import ui

ui.hero("Detect / Seizure Hunter",
        "Human <em>versus</em> model",
        "Neurologists train for years to read EEG. Each round is one minute from a held-out recording; call the "
        "seizure as early as you can without raising false alarms, and compare your timing with the three detectors.",
        index="05", meta="ROUNDS <b>6</b><br>SEIZURE ROUNDS <b>4</b><br>CONTROL ROUNDS <b>2</b>")
ui.need_artifacts()
ui.component("game", {"rounds": ui.rounds()}, height=740)
st.caption("Scoring: a detected seizure earns up to 100 points, minus 4 per second of delay. A false call costs 40. "
           "Staying silent on a seizure-free clip earns 60. Models are scored on their first alarm, using the same "
           "rule as the Live Monitor.")
