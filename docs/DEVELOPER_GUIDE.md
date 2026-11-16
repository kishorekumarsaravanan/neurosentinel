# NeuroSentinel developer guide

## 1. Setup

```bash
pip install -r requirements-dev.txt      # app dependencies + pytest
bash scripts/download_data.sh            # optional: raw EEG for training and the data-dependent pages
python train.py                          # optional: retrain; rewrites artifacts/
streamlit run streamlit_app.py
pytest -v                                # run the test suite
```

`scikit-learn` is pinned to the version the committed models were pickled with. If you change it, retrain and commit
the new `artifacts/`.

## 2. Architecture

```
            data/raw/*.edf  (CHB-MIT, downloaded)
                  │
   neurosentinel/ │  edf.py ─► data.py ─► preprocess.py ─► features.py ─► models.py ─► evaluate.py
                  │
   train.py  ─────┴─► artifacts/  (models, results.json, predictions/*.npz, demo/*.json)
                                    │
   streamlit_app.py ─► nsapp/ui.py ─┴─► nsapp/views/*.py ─► nsapp/components/*.html (Canvas / WebAudio)
```

| Path | Responsibility |
|---|---|
| `neurosentinel/config.py` | Paths, montage (18 channels), signal parameters, patient train/test split, alarm rule |
| `neurosentinel/edf.py` | Dependency-free EDF reader; `select_channels` picks the montage by label |
| `neurosentinel/data.py` | Seizure annotations from `<patient>-summary.txt`, loading and masking recordings |
| `neurosentinel/preprocess.py` | 60 Hz notch, zero-phase 0.5–40 Hz band-pass, windowing, seizure-overlap labels |
| `neurosentinel/features.py` | 21 features per channel (8 time-domain, 13 frequency-domain), 378 in total |
| `neurosentinel/models.py` | SVM and random-forest factories, the 1D-CNN and its training wrapper |
| `neurosentinel/evaluate.py` | Window metrics, K-of-N alarm smoothing, event-level metrics |
| `train.py` | Builds training sets, trains all models, streams held-out hours, exports app assets |
| `streamlit_app.py` | Page config, navigation, sidebar, theme toggle |
| `nsapp/ui.py` | Design tokens (light/dark), CSS, Plotly styling, component bridge, cached data loaders |
| `nsapp/views/` | One file per page |
| `nsapp/components/` | Self-contained HTML/JS widgets: `monitor`, `game`, `hypersync`, `brainmap`; `shared.js/css` and `icons.js` are injected into each |
| `static/fonts/` | Self-hosted fonts served by Streamlit static serving |
| `tests/` | pytest suite |

## 3. Data flow

1. **Training set** (`train.build_train_set`): seizure windows with 1 s stride (≥ 90 % inside a seizure) and
   non-overlapping background windows (no seizure overlap), undersampled to 8 background windows per seizure window.
2. **Models**: SVM and random forest on the 378 features; the CNN on the filtered signal, normalised per channel by the
   background standard deviation. The CNN keeps the checkpoint with the lowest validation loss.
3. **Held-out evaluation** (`train.stream_record`): every test hour is scored at 1 s stride. Window metrics use every
   4th window (non-overlapping); event metrics use the K-of-N alarm on the 1 Hz output.
4. **Exports**: `results.json` (metrics, curves, histories, RF importances), `predictions/<record>.npz` (per-second
   probabilities, labels, band powers), `demo/monitor_clips.json` and `demo/game_rounds.json` (EEG as base64 int16 at
   128 Hz plus probabilities).

To add a patient, download its files, add an entry to `PATIENTS` in `config.py` (disjoint train and test
recordings, test set with at least one seizure-free hour) and rerun `train.py`.

## 4. The app

### Theme tokens
All colours come from `TOKENS` in `nsapp/ui.py` (`light` = IBM Carbon on paper, `dark` = Carbon g100). The module
defines `__getattr__`, so `ui.INK`, `ui.SEIZURE`, `ui.MODEL_COLORS`, `ui.ACTIVITY` always return the value for the
viewer's current mode. **Never hard-code colours in a view**; add a token instead.

Components receive the mode as `<html data-theme="…">`; `shared.css` overrides the CSS variables and `shared.js`
selects `NS.C` (canvas colours) and the colour maps.

### Theme switching
`ui.theme_toggle()` stores the choice in `st.session_state["ns_theme"]`, and `ui.theme_bridge()` writes the `ns_theme`
cookie plus Streamlit's cached theme (`stActiveTheme-<path>-v1` = `"Light"`/`"Dark"` in localStorage, mapped to
`[theme.light]`/`[theme.dark]` in `.streamlit/config.toml`) and reloads when they disagree. A guard prevents more than
two reloads in 15 s.

### Adding a page
1. Create `nsapp/views/<name>.py` starting with `ui.hero(section, title, subtitle, index=..., meta=...)`.
2. Register it in `streamlit_app.py` with a Material Symbols icon (`:material/<name>:`).
3. Add it to `PAGE_PATHS` in `nsapp/ui.py` so the theme is kept for direct links to the page.
4. Add it to `ARTIFACT_PAGES` or `DATA_PAGES` in `tests/test_app.py`.

### Adding a component
Write `nsapp/components/<name>.html` with the placeholders `/*__CSS__*/`, `/*__JS__*/` and `__DATA__`, then call
`ui.component("<name>", data_dict, height)`. Use `NS.C` for colours and `NS.icon(name)` for Lucide icons (add new
icons to `icons.js`).

### Gotchas
- Pass `theme=None` to `st.plotly_chart`; Streamlit's Plotly theme overrides our styling.
- Streamlit's Plotly renderer replaces exact colours `#000001`…`#00000N` with theme colours; avoid them in scales.
- A `st.cache_data` function whose output embeds colours must take those colours as arguments.
- Canvases inside components must be absolutely positioned (`canvas.cv`), or they grow their grid rows.
- Use `width="stretch"` instead of the removed `use_container_width`.

## 5. Tests

```bash
pytest -v                # everything available locally
pytest -m "not app"      # unit tests only (seconds)
pytest -m app            # Streamlit smoke tests
```

| File | Covers |
|---|---|
| `test_edf.py` | EDF round trip, scaling, montage selection, missing channels |
| `test_preprocess.py` | Filter pass/stop bands, notch, zero phase, windowing, seizure overlap |
| `test_features.py` | Feature inventory, shape, chunking, physiology (alpha peak, ictal amplitude/entropy) |
| `test_evaluate.py` | Confusion-based metrics, accuracy paradox, K-of-N alarms, refractory merging, latency and false alarms |
| `test_models.py` | CNN shape and size, all models learn a separable toy problem, CNN save/load, saliency |
| `test_data.py` | Summary parsing (both annotation formats), masks, disjoint splits, a real recording |
| `test_app.py` | Every page renders in light and dark mode; scope switching; a full Diagnose scan |

Tests that need the raw EEG are skipped automatically when `data/raw` is empty, so CI (GitHub Actions,
`.github/workflows/tests.yml`) runs the unit tests and the artifact-based app tests on every pull request.

## 6. Workflow

1. Create a branch from `main`: `git switch -c <short-description>`.
2. Commit focused changes; run `pytest` before pushing.
3. Push and open a pull request; CI must pass and a teammate reviews.
4. Merge into `main` and delete the branch.
