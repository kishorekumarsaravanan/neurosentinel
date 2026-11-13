"""Design system, component bridge and cached data access for the NeuroSentinel app.

Design tokens (two modes, switched from the sidebar)
  light   : IBM Carbon tokens (gray 10-100, blue 60, red 60, yellow 60, green 60) on a warm paper base
  dark    : IBM Carbon g100 theme (gray 100/90/80, blue 40, red 50, yellow 30, green 40)
  data    : Carbon categorical palette (purple, teal, cyan), one step lighter in dark mode for contrast
  scales  : light = ColorBrewer (YlOrRd, BuPu, YlGnBu, RdBu); dark = matplotlib perceptual maps (inferno,
            viridis, magma, berlin) whose dark ends sit naturally on a dark background
  type    : Newsreader (display serif), IBM Plex Sans (UI), IBM Plex Mono (figures), self-hosted in ./static
  icons   : Material Symbols (Streamlit chrome), Lucide (custom components and HTML blocks)

Tokens are resolved per mode at attribute-access time (module __getattr__), so `ui.INK`, `ui.SEIZURE`,
`ui.MODEL_COLORS`, `ui.ACTIVITY` ... always match the theme the viewer selected.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from neurosentinel.config import (ARTIFACTS, PRED_DIR, DEMO_DIR, MODEL_NAMES, CHANNELS, FS, PATIENTS)

COMP_DIR = Path(__file__).parent / "components"
ICON = {"svm": "chart-scatter", "rf": "trees", "cnn": "network"}

# ----------------------------------------------------------------------------- palettes
# ColorBrewer (Cynthia Brewer, colorbrewer2.org)
YLORRD = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"]
BUPU = ["#f7fcfd", "#e0ecf4", "#bfd3e6", "#9ebcda", "#8c96c6", "#8c6bb1", "#88419d", "#810f7c", "#4d004b"]
YLGNBU = ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#253494", "#081d58"]
RDBU_R = ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
# matplotlib perceptual colormaps (van der Walt & Smith; Crameri for berlin), sampled at 9 stops.
# First stop nudged from #000004: Streamlit treats #00000N as theme placeholders inside Plotly figures.
INFERNO = ["#05040f", "#1f0c48", "#540f6d", "#87216b", "#b73557", "#e05536", "#f8870e", "#fac228", "#f5f992"]
VIRIDIS = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"]
MAGMA = ["#05040f", "#1c1044", "#4e117b", "#802582", "#b2357b", "#e24d66", "#fa815f", "#febb81", "#fcf4b6"]
BERLIN = ["#9eb0ff", "#519fd3", "#286886", "#14303e", "#190c09", "#411201", "#7d341e", "#be6f63", "#ffadad"]

TOKENS = {
    "light": dict(
        PAPER="#F6F5F1", SURFACE="#FFFFFF", INSET="#ECEAE4", LINE="#DCD9D1", LINE_STRONG="#B9B5AA",
        INK="#161616", INK2="#525252", INK3="#8D8D8D", ACCENT="#0F62FE", ACCENT_SOFT="#78A9FF",
        SEIZURE="#DA1E28", WARN="#B28600", OK="#198038",
        GRID="#EEECE7", GRID_STRONG="#E8E6E0", FILL_MUTED="#C6C3BA", FILL_MUTED2="#B9B5AA",
        MODEL_COLORS={"svm": "#6929C4", "rf": "#005D5D", "cnn": "#1192E8"},
        ACTIVITY=YLORRD, IMPORTANCE=["#ffffff"] + BUPU[2:], SPECTRO=YLGNBU, DIVERGING=RDBU_R,
        BANDS_COL=["#08519c", "#3182bd", "#6baed6", "#fd8d3c", "#d94801"],
        SEQ_SOFT="#78A9FF",
    ),
    "dark": dict(
        PAPER="#161616", SURFACE="#1F1F1F", INSET="#2E2E2E", LINE="#393939", LINE_STRONG="#6F6F6F",
        INK="#F4F4F4", INK2="#C6C6C6", INK3="#A8A8A8", ACCENT="#78A9FF", ACCENT_SOFT="#A6C8FF",
        SEIZURE="#FA4D56", WARN="#F1C21B", OK="#42BE65",
        GRID="#2A2A2A", GRID_STRONG="#333333", FILL_MUTED="#525252", FILL_MUTED2="#6F6F6F",
        MODEL_COLORS={"svm": "#A56EFF", "rf": "#08BDBA", "cnn": "#33B1FF"},
        ACTIVITY=INFERNO, IMPORTANCE=VIRIDIS, SPECTRO=MAGMA, DIVERGING=BERLIN,
        BANDS_COL=["#4589ff", "#78a9ff", "#a6c8ff", "#ffb784", "#ff832b"],
        SEQ_SOFT="#A6C8FF",
    ),
}


def mode() -> str:
    """Active theme: session choice, else the cookie written by the toggle, else light."""
    if "ns_theme" not in st.session_state:
        try:
            c = st.context.cookies.get("ns_theme")
        except Exception:
            c = None
        st.session_state["ns_theme"] = c if c in TOKENS else "light"
    return st.session_state["ns_theme"]


def T() -> dict:
    return TOKENS[mode()]


def __getattr__(name: str):          # PEP 562: ui.INK, ui.MODEL_COLORS ... resolve per mode
    tok = TOKENS["light"]
    if name in tok:
        return T()[name]
    if name == "MUTED":
        return T()["INK2"]
    raise AttributeError(name)


def scale(colors: list[str]) -> list[list]:
    return [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]


# ----------------------------------------------------------------------------- icons (Lucide, bundled)
_ICONS = json.loads(re.search(r"const ICONS = (\{.*?\});", (COMP_DIR / "icons.js").read_text(encoding="utf-8"), re.S).group(1))


def icon(name: str, size: int = 16, color: str = "currentColor", stroke: float = 1.75) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align:-0.18em">{_ICONS.get(name, "")}</svg>')


# ----------------------------------------------------------------------------- CSS
CSS = """
<style>
__ROOT__
.stApp { background: var(--paper); }
html, body, .stMarkdown, p, li, label, input, textarea, button { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'Newsreader', Georgia, serif !important; font-weight: 500 !important; letter-spacing: -0.01em; color: var(--ink); }
h3 { font-size: 1.55rem !important; } h4 { font-size: 1.25rem !important; } h5 { font-family: 'IBM Plex Sans'; font-weight: 600; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1400px; }
section[data-testid="stSidebar"] { border-right: 1px solid var(--sidebar-edge); }
[data-testid="stSidebarNav"] a span:not([data-testid="stIconMaterial"]) { font-family: 'IBM Plex Sans'; font-size: 14px; letter-spacing: .005em; }
[data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"] { font-family: 'IBM Plex Mono'; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: #A8A8A8; }
img[data-testid="stLogo"] { height: 2.6rem !important; max-width: 100% !important; }

/* page header */
.ns-hero { padding: 0 0 22px; margin-bottom: 26px; border-bottom: 1px solid var(--ink); display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: end; }
.ns-eyebrow { font-family: 'IBM Plex Mono'; font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--ink2); display: flex; gap: 12px; align-items: center; }
.ns-eyebrow .idx { color: var(--accent); }
.ns-eyebrow .rule { width: 28px; height: 1px; background: var(--ink2); display: inline-block; }
.ns-title { font-family: 'Newsreader', Georgia, serif; font-weight: 450; font-size: 52px; line-height: 1.02; margin: 12px 0 14px; color: var(--ink); letter-spacing: -0.02em; }
.ns-title em { font-style: italic; color: var(--accent); }
.ns-sub { color: var(--ink2); font-size: 16.5px; max-width: 820px; line-height: 1.6; }
.ns-sub b { color: var(--ink); font-weight: 600; }
.ns-meta { font-family: 'IBM Plex Mono'; font-size: 11.5px; color: var(--ink3); text-align: right; line-height: 1.8; white-space: nowrap; }
.ns-meta b { color: var(--ink); font-weight: 500; }

/* stat tiles */
.ns-cards { display: grid; gap: 0; margin: 4px 0 18px; border: 1px solid var(--line); background: var(--surface); }
.ns-card { padding: 16px 18px 15px; border-right: 1px solid var(--line); position: relative; }
.ns-card:last-child { border-right: 0; }
.ns-card .l { font-size: 12px; letter-spacing: .04em; color: var(--ink2); display: flex; align-items: center; gap: 7px; }
.ns-card .l i { width: 8px; height: 8px; display: inline-block; background: var(--acc); }
.ns-card .v { font-family: 'IBM Plex Mono'; font-size: 28px; font-weight: 500; margin-top: 6px; color: var(--ink); letter-spacing: -0.02em; }
.ns-card .s { color: var(--ink3); font-size: 12.5px; margin-top: 2px; }

.ns-panel { padding: 20px 22px; border: 1px solid var(--line); background: var(--surface); }
.ns-step { display: grid; grid-template-columns: 46px 1fr; gap: 14px; align-items: start; padding: 14px 0; border-bottom: 1px solid var(--line); }
.ns-step:last-child { border-bottom: 0; }
.ns-step .n { font-family: 'IBM Plex Mono'; color: var(--accent); font-size: 13px; padding-top: 2px; }
.ns-step b { font-family: 'Newsreader'; font-size: 19px; font-weight: 500; }
.ns-step div div { color: var(--ink2); font-size: 14px; line-height: 1.55; margin-top: 2px; }
.ns-links a { display: grid; grid-template-columns: 22px 1fr auto; gap: 10px; align-items: center; padding: 11px 0; border-bottom: 1px solid var(--line);
   color: var(--ink) !important; text-decoration: none !important; font-size: 14.5px; }
.ns-links a span.d { color: var(--ink2); font-size: 13px; display: block; }
.ns-links a:hover { color: var(--accent) !important; }
.ns-note { border-left: 2px solid var(--accent); padding: 8px 16px; color: var(--ink2); background: var(--surface); font-size: 14px; }
.ns-note b { color: var(--ink); }
.ns-label { font-family: 'IBM Plex Mono'; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--ink2); }

/* model cards */
.ns-model { background: var(--surface); border: 1px solid var(--line); border-top: 3px solid var(--acc); padding: 18px 20px 16px; position: relative; height: 100%; }
.ns-model .head { display: flex; align-items: center; gap: 12px; }
.ns-model .ico { width: 40px; height: 40px; border: 1px solid var(--acc); color: var(--acc); display: grid; place-items: center; border-radius: 50%; }
.ns-model .nm { font-family: 'Newsreader'; font-size: 25px; font-weight: 500; line-height: 1.1; }
.ns-model .cls { font-family: 'IBM Plex Mono'; font-size: 10.5px; color: var(--ink3); letter-spacing: .1em; text-transform: uppercase; margin-top: 3px; }
.ns-model .stat { display: grid; grid-template-columns: 1fr auto; font-size: 13.5px; margin-top: 10px; color: var(--ink2); }
.ns-model .stat b { font-family: 'IBM Plex Mono'; font-weight: 500; color: var(--ink); }
.ns-model .bar { grid-column: 1 / 3; height: 3px; background: var(--inset); margin-top: 5px; }
.ns-model .bar i { display: block; height: 100%; background: var(--acc); }
.ns-model .badge { display: inline-block; margin: 0 0 12px; font-family: 'IBM Plex Mono'; font-size: 10.5px; letter-spacing: .08em;
   padding: 3px 8px; border: 1px solid var(--acc); color: var(--acc); text-transform: uppercase; }
.ns-model .kpis { margin-top: 16px; border-top: 1px solid var(--ink); padding-top: 4px; }
.ns-model .kpis .kr { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; border-bottom: 1px solid var(--line); font-size: 13px; color: var(--ink2); }
.ns-model .kpis .kr b { font-family: 'IBM Plex Mono'; font-weight: 500; color: var(--ink); white-space: nowrap; }
.modebar-container { display: none !important; }
.ns-model .foot { font-size: 12px; color: var(--ink3); margin-top: 12px; }
.ns-model .quip { margin-top: 12px; font-family: 'Newsreader'; font-style: italic; font-size: 15px; color: var(--ink2); }

div[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); padding: 10px 14px; }
.stTabs [data-baseweb="tab-list"] { gap: 22px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Sans'; font-size: 14px; padding: 8px 0; background: transparent; }
.stTabs [aria-selected="true"] { color: var(--ink); }
[data-testid="stCaptionContainer"] { color: var(--ink2); }
iframe { border: 1px solid var(--line); background: var(--surface); }
</style>
"""


def setup():
    t = T()
    root = (f":root {{ --paper:{t['PAPER']}; --surface:{t['SURFACE']}; --inset:{t['INSET']}; --line:{t['LINE']}; "
            f"--line2:{t['LINE_STRONG']}; --ink:{t['INK']}; --ink2:{t['INK2']}; --ink3:{t['INK3']}; --accent:{t['ACCENT']}; "
            f"--seizure:{t['SEIZURE']}; --warn:{t['WARN']}; --sidebar-edge:{'#262626' if mode() == 'dark' else 'transparent'}; "
            f"color-scheme:{mode()}; }}")
    st.markdown(CSS.replace('__ROOT__', root), unsafe_allow_html=True)


def hero(section: str, title: str, sub: str, index: str = "", meta: str = ""):
    idx = f'<span class="idx">{index}</span><span class="rule"></span>' if index else ""
    meta_html = f'<div class="ns-meta">{meta}</div>' if meta else "<div></div>"
    st.markdown(f'<div class="ns-hero"><div><div class="ns-eyebrow">{idx}{section}</div><div class="ns-title">{title}</div>'
                f'<div class="ns-sub">{sub}</div></div>{meta_html}</div>', unsafe_allow_html=True)


def cards(items: list[tuple], cols: int | None = None):
    cols = cols or len(items)
    ink = T()["INK"]
    html = "".join(f'<div class="ns-card" style="--acc:{c or ink}"><div class="l"><i></i>{l}</div><div class="v">{v}</div>'
                   f'<div class="s">{s}</div></div>' for l, v, s, c in items)
    st.markdown(f'<div class="ns-cards" style="grid-template-columns:repeat({cols},1fr)">{html}</div>', unsafe_allow_html=True)


def panel(html: str):
    st.markdown(f'<div class="ns-panel">{html}</div>', unsafe_allow_html=True)


def note(html: str):
    st.markdown(f'<div class="ns-note">{html}</div>', unsafe_allow_html=True)


def component(name: str, data: dict, height: int):
    html = (COMP_DIR / f"{name}.html").read_text(encoding="utf-8").replace("<html>", f'<html data-theme="{mode()}">', 1)
    html = html.replace("/*__CSS__*/", (COMP_DIR / "shared.css").read_text(encoding="utf-8"))
    html = html.replace("/*__JS__*/", (COMP_DIR / "icons.js").read_text(encoding="utf-8") + "\n" +
                        (COMP_DIR / "shared.js").read_text(encoding="utf-8"))
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    components.html(html, height=height, scrolling=False)


# ----------------------------------------------------------------------------- plotly styling
def style(fig: go.Figure, height: int = 380, legend: bool = True, margin=None) -> go.Figure:
    t = T()
    fig.update_layout(
        template="simple_white", height=height, paper_bgcolor=t["SURFACE"], plot_bgcolor=t["SURFACE"],
        font=dict(family="IBM Plex Sans, sans-serif", color=t["INK2"], size=12),
        margin=margin or dict(l=14, r=14, t=62, b=12), showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.0, yanchor="bottom", x=1, xanchor="right", font=dict(size=11.5, color=t["INK2"])),
        hoverlabel=dict(bgcolor=t["INK"], bordercolor=t["INK"], font=dict(family="IBM Plex Mono", color=t["PAPER"], size=12)),
        title_font=dict(family="Newsreader, Georgia, serif", size=19, color=t["INK"]), title_x=0.01,
        polar=dict(bgcolor=t["SURFACE"]),
    )
    if fig.layout.title.text:
        fig.update_layout(title_y=0.975, title_yanchor="top")
    else:
        fig.update_layout(title_text="")     # avoids plotly rendering the literal word "undefined"
    axis = dict(gridcolor=t["GRID"], zerolinecolor=t["LINE"], linecolor=t["LINE_STRONG"], tickcolor=t["LINE_STRONG"],
                tickfont=dict(family="IBM Plex Mono", size=10.5, color=t["INK2"]), title_font=dict(size=12, color=t["INK2"]))
    fig.update_xaxes(**axis, showgrid=True)
    fig.update_yaxes(**axis, showgrid=True)
    return fig


# ----------------------------------------------------------------------------- data access
def artifacts_ready() -> bool:
    return (ARTIFACTS / "results.json").exists()


def need_artifacts():
    if not artifacts_ready():
        st.warning("Trained models not found. Run `python train.py` once to build `artifacts/` "
                   "(models, predictions, demo clips).", icon=":material/model_training:")
        st.stop()


@st.cache_data(show_spinner=False)
def results() -> dict:
    return json.loads((ARTIFACTS / "results.json").read_text())


@st.cache_data(show_spinner=False)
def clips() -> list[dict]:
    return json.loads((DEMO_DIR / "monitor_clips.json").read_text())


@st.cache_data(show_spinner=False)
def rounds() -> list[dict]:
    return json.loads((DEMO_DIR / "game_rounds.json").read_text())


@st.cache_data(show_spinner=False)
def prediction(record: str) -> dict:
    z = np.load(PRED_DIR / f"{record}.npz")
    return {k: z[k] for k in z.files}


@st.cache_resource(show_spinner=False)
def models(patient: str):
    from neurosentinel.models import PatientModels
    return PatientModels(patient)


@st.cache_data(show_spinner=False, max_entries=4)
def raw_signal(record: str) -> np.ndarray:
    from neurosentinel.data import load_raw
    return load_raw(record)


@st.cache_data(show_spinner=False, max_entries=4)
def clean_signal(record: str) -> np.ndarray:
    from neurosentinel.preprocess import preprocess
    return preprocess(raw_signal(record))


def test_records() -> list[str]:
    return [r for p in PATIENTS.values() for r in p["test"]]


def record_label(rec: str) -> str:
    from neurosentinel.data import seizures_of
    n = len(seizures_of(rec))
    tag = "held-out" if rec in test_records() else "training"
    return f"{rec}  ·  {n} seizure{'s' if n != 1 else ''}  ·  {tag}"


def hexa(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def model_label(k: str, size: int = 14) -> str:
    c = T()['MODEL_COLORS'][k]
    return f'<span style="color:{c}">{icon(ICON[k], size, c)}</span> {MODEL_NAMES[k]}'


# ----------------------------------------------------------------------------- theme switching
PAGE_PATHS = ["/", "/atlas", "/signal_lab", "/monitor", "/game", "/diagnose", "/arena", "/xray"]


def theme_bridge(force_reload: bool = False):
    """Keep Streamlit's own widget theme (persisted per path in localStorage) in step with our mode.

    Writes the `ns_theme` cookie (read by Python on the next session) and Streamlit's cached preset
    ("Light"/"Dark", merged with [theme.light]/[theme.dark] in config.toml), then reloads only when the
    theme Streamlit actually applied differs from the one requested.
    """
    want = "Dark" if mode() == "dark" else "Light"
    js = f"""<script>
(function () {{
  try {{
    const want = "{want}", paths = {json.dumps(PAGE_PATHS)}.concat([window.location.pathname]);
    const key = p => "stActiveTheme-" + p + "-v1";
    let prev = null;
    try {{ prev = JSON.parse(localStorage.getItem(key(window.location.pathname)) || "null"); }} catch (e) {{}}
    const sys = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "Dark" : "Light";
    // Streamlit stores presets as "Light"/"Dark" or, with [theme.light]/[theme.dark], "Custom Theme Light/Dark"
    const nm = prev && prev.name ? prev.name : "";
    const applied = /Auto|system/i.test(nm) || !nm ? sys : (/Dark/.test(nm) ? "Dark" : "Light");
    document.cookie = "ns_theme={mode()}; path=/; max-age=31536000; SameSite=Lax";
    paths.forEach(p => localStorage.setItem(key(p), JSON.stringify({{ name: want }})));
    // guard: never reload more than twice within 15 s
    const now = Date.now(), log = JSON.parse(sessionStorage.getItem("ns_reloads") || "[]").filter(t => now - t < 15000);
    if (({str(force_reload).lower()} || applied !== want) && log.length < 2) {{
      sessionStorage.setItem("ns_reloads", JSON.stringify(log.concat([now])));
      window.location.reload();
    }}
  }} catch (e) {{ console.warn("theme bridge", e); }}
}})();
</script>"""
    st.html(js, unsafe_allow_javascript=True)


def theme_toggle():
    """Sidebar switch. Flipping it updates the mode, then reloads so Streamlit's widgets follow."""
    dark = st.toggle("Dark mode", value=mode() == "dark", key="ns_dark_toggle")
    new = "dark" if dark else "light"
    if new != mode():
        st.session_state["ns_theme"] = new
        theme_bridge(force_reload=True)
        st.stop()
    theme_bridge()
