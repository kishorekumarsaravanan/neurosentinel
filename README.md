# NeuroSentinel: EEG seizure detection

Binary EEG classification (seizure vs non-seizure) on the **CHB-MIT Scalp EEG** dataset, comparing a traditional,
an ensemble and a deep-learning model, presented through an interactive Streamlit app.

| Category | Model | Input |
|---|---|---|
| Traditional ML | SVM (RBF, class-weighted) | 378 hand-crafted features |
| Ensemble | Random Forest (300 trees) | 378 hand-crafted features |
| Deep learning | 1D-CNN (≈110 k params) | filtered raw EEG, 18 ch × 1024 samples |

## Quick start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py     # trained models ship in artifacts/, so the app runs immediately
```

Optional:

```bash
bash scripts/download_data.sh      # ~890 MB raw EEG (chb01 + chb03): needed by Brain Atlas, Signal Lab, Explainability, Diagnose
python train.py                    # ~7 min on CPU: retrain, re-evaluate, rewrite artifacts/
```

## Documentation

- [User guide](docs/USER_GUIDE.md): installation, every page step by step, metrics, troubleshooting
- [Developer guide](docs/DEVELOPER_GUIDE.md): architecture, data flow, theming, adding pages/components, tests, workflow

## The app

| Section | Page | What it does |
|---|---|---|
| Discover | **Overview** | Project summary, dot-matrix EEG figure, Kuramoto-oscillator simulation of hypersynchrony |
| | **Brain Atlas** | Channel-activity map of a full hour, animated scalp time-lapse of the seizure spreading, 18-channel EEG pages |
| | **Signal Lab** | Interactive filtering (band-pass, notch, order), PSD, spectrogram, windowing, per-window feature vector |
| Detect | **Live Monitor** | Clinical review-station replay of held-out seizures: scrolling EEG, scalp map, three model cards, consensus alarm, assessment line, sonification |
| | **Seizure Hunter** | Game: spot seizures in real EEG faster than the three models; scored and ranked |
| | **Diagnose** | Upload an EDF → per-second model output → detected events → downloadable HTML report |
| Evaluate | **Model Comparison** | Model cards, performance profile, scorecard, confusion matrices, ROC/PR, detection latency, accuracy paradox, alarm tuning, CNN training curves |
| | **Explainability** | Occlusion head maps per model, CNN gradient × input saliency, random-forest feature importance |

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

47 tests: unit tests for the EDF reader, filters, features, models, metrics and alarm logic, plus Streamlit smoke
tests that open every page in light and dark mode. Tests that need the raw EEG are skipped when `data/raw` is empty.
GitHub Actions runs the suite on every pull request (`.github/workflows/tests.yml`).

## Method

1. **Data**: CHB-MIT (PhysioNet), 256 Hz, 18 bipolar channels of the double-banana montage (common to all patients).
2. **Preprocessing**: 60 Hz notch, 4th-order Butterworth band-pass 0.5–40 Hz (zero-phase).
3. **Segmentation**: 4 s windows. Training uses seizure windows with 1 s stride (75 % overlap augmentation) and
   non-overlapping background windows, undersampled to 8:1. A window counts as seizure if ≥ 50 % of it is ictal.
4. **Features (21 per channel)**:
   - Time domain: log-variance, skewness, kurtosis, line length, zero-crossing rate, Hjorth mobility and complexity, peak-to-peak.
   - Frequency domain: absolute and relative δ θ α β γ power, spectral entropy, peak frequency, 90 % spectral edge.
5. **Validation**: patient-specific; training and test recordings are disjoint. Each patient's test set contains 2 unseen
   seizures plus one entirely seizure-free hour, and is streamed at 1 s stride.
6. **Metrics**:
   - Window level: Accuracy, Precision, Recall (Sensitivity), Specificity, F1, ROC-AUC, PR-AUC, confusion matrix.
   - Event level: seizures detected, detection latency, false alarms per hour (alarm = 3 of the last 4 s positive).

| Patient | Train recordings | Test recordings (unseen) |
|---|---|---|
| chb01 | 01, 02, 03, 04, 05, 15, 16, 18 (5 seizures) | 06 (seizure-free), 21, 26 |
| chb03 | 01, 02, 03, 04, 05, 06, 34 (5 seizures) | 07 (seizure-free), 35, 36 |

## Design system

| Layer | Choice | Source |
|---|---|---|
| UI colours | Gray 10–100, Blue 60 (interactive), Red 60 (seizure), Yellow 60 (threshold) on a warm paper base | IBM Carbon Design System |
| Model colours | Purple 70 (SVM), Teal 70 (Random Forest), Cyan 50 (1D-CNN) | Carbon categorical data-vis palette |
| Heat maps | YlOrRd (activity), BuPu (importance), YlGnBu (spectrogram), RdBu (z-scores) | ColorBrewer (C. Brewer) |
| Dark mode | Carbon g100 theme (gray 100/90/80, blue 40, red 50); models in Carbon purple/teal/cyan one step lighter; heat maps switch to inferno, viridis, magma and berlin | IBM Carbon; matplotlib perceptual colormaps |
| Type | Newsreader (headings), IBM Plex Sans (UI), IBM Plex Mono (figures) | Google Fonts, SIL Open Font License, self-hosted in `static/fonts` |
| Icons | Material Symbols (navigation, buttons), Lucide (custom components) | Google (Apache 2.0); lucide.dev (ISC), bundled in `nsapp/components/icons.js` |

Everything is served locally, so the app runs without an internet connection during a demo.

**Light / dark mode:** use the *Dark mode* switch in the sidebar. The choice is remembered in the browser (cookie `ns_theme`) and applied to Streamlit's own widgets (via `[theme.light]` / `[theme.dark]` in `.streamlit/config.toml`), the page styling, every chart and the custom components.

## Layout

```
neurosentinel/   core library: EDF reader, preprocessing, features, models, evaluation
train.py         end-to-end training + evaluation + export of app assets
streamlit_app.py app entry point (navigation, sidebar)
nsapp/ui.py      theme, component bridge, cached loaders
nsapp/views/     one file per page
nsapp/components custom HTML/Canvas/WebAudio components (monitor, game, hypersync, brainmap) + icons.js
nsapp/assets     logo and mark (wordmark drawn from Newsreader glyph outlines)
static/fonts     self-hosted Newsreader, IBM Plex Sans, IBM Plex Mono
artifacts/       generated: models, results.json, predictions, demo clips
tests/           pytest suite (unit + Streamlit smoke tests)
docs/            user guide and developer guide
.github/         CI workflow (pytest on every pull request)
```

## Contributing

Work on a branch, run `pytest`, open a pull request; CI must pass and a teammate reviews before merging into `main`.
See the [developer guide](docs/DEVELOPER_GUIDE.md#6-workflow).

*Research prototype for a BCI course project. Not a medical device.*
