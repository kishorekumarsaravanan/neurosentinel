# NeuroSentinel user guide

NeuroSentinel compares three seizure detectors (an SVM, a random forest and a 1D-CNN) on scalp EEG from the
CHB-MIT database and lets you explore the data, the models and their decisions interactively.

> Research prototype for a BCI course project. It is **not a medical device** and must not be used for diagnosis.

## 1. Installation

Requirements: Python 3.11 or newer, about 2 GB of free disk space, any recent browser.

```bash
git clone https://github.com/kishorekumarsaravanan/neurosentinel.git
cd neurosentinel
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at <http://localhost:8501>. The trained models and demo clips ship with the repository, so the
**Live Monitor**, **Seizure Hunter** and **Model Comparison** pages work immediately.

The **Brain Atlas**, **Signal Lab**, **Explainability** and **Diagnose** pages read raw EEG. Download it once:

```bash
bash scripts/download_data.sh      # ~890 MB: 21 one-hour recordings of patients chb01 and chb03
```

On Windows, run the script from Git Bash.

## 2. Interface

- **Sidebar**: page navigation in three groups (Discover, Detect, Evaluate), the **Dark mode** switch and a status
  block. Your theme choice is remembered by the browser.
- **Page header**: section, title, a short explanation, and key facts on the right.
- Every chart supports hover for exact values. Most controls (dropdowns, sliders, switches) update the page instantly.

## 3. Pages

### Overview
A summary of the project, a dot-matrix figure of one minute of real EEG (the seizure is shown in red), and a
simulation of 160 coupled neurons.
- Drag **Coupling K** past about 3.5, or press **Trigger seizure**, to watch the neurons synchronise and the simulated
  EEG turn into large rhythmic waves. **Reset** returns to the asynchronous state.

### Brain Atlas
One hour of EEG at a glance.
1. Choose a **Recording**. Held-out recordings also show the three models' output.
2. The top chart shows activity per channel over time; seizures appear as bright vertical bands.
3. **Scalp time-lapse**: choose what to colour the head by (activity or a frequency band) and a time range, then press
   play. Red frames on the timeline are inside the annotated seizure. Click the timeline to jump.
4. **EEG page view**: a 20-second page of all 18 channels. Move **Start** to scroll, **Amplitude** to zoom.

### Signal Lab
The preprocessing pipeline, step by step.
1. Choose a recording, a channel and a focus time.
2. **Filtering**: adjust the high-pass, low-pass, filter order and 60 Hz notch; compare raw and filtered signals and
   their power spectra.
3. **Time–frequency view**: spectrogram of 60 s around the focus.
4. **Segmentation**: how the signal is cut into 4-second windows (red = labelled seizure).
5. **Feature vector**: how unusual the current window is compared with the patient's background, for one channel
   (polar chart) and all channels (heat map).

### Live Monitor
A replay of a held-out recording on a clinical-style review station.
1. Choose a **Recording** (clips marked "contains a seizure" or "seizure-free control") and an initial speed.
2. Press **Play** (or the space bar). Use **Jump to pre-onset** to skip to 25 s before the seizure.
3. Read the display:
   - EEG traces scroll from right to left; the red band is the neurologist's annotation (toggle **Annotation**).
   - The three model cards show each model's seizure probability; a card turns red when that model raises an alarm.
   - **Consensus**: the alarm (red status bar) fires when two of three models are in alarm.
   - **Scalp activity** shows where amplitude is rising; the **Assessment** line summarises the state.
   - The **Event log** lists onsets, alarms and their latency.
   - The bar at the bottom is the whole clip; click it to seek.
4. **Sound** turns the signal envelope into audio and plays an alarm tone. **Gain** changes the trace amplitude.
5. Below the monitor, the full-hour chart shows every model's output and alarms across the recording.

### Seizure Hunter
A game: spot seizures faster than the models.
1. Pick a playback speed and press **Start**.
2. Each round plays one minute of held-out EEG. Press **Space** (or **Seizure**) as soon as you think a seizure has
   started. You get one call per round.
3. After each round the answer is revealed with your call and each model's first alarm.
4. Scoring: up to 100 points for a detection (minus 4 per second of delay), −40 for a false call, +60 for staying
   silent on a seizure-free clip. The final screen ranks you against the three models.

### Diagnose
Screen your own recording.
1. **Upload an EDF recording** (CHB-MIT format, 256 Hz, containing the 18 bipolar channels) or choose one from the
   library.
2. Choose the **patient-specific model set** (detectors are trained per patient).
3. Press **Scan recording**. A one-hour file takes a few seconds.
4. Read the results: summary tiles, the per-second output of each model, detected events (with the region of
   strongest activity and, for known files, whether they match an annotation).
5. **Download report (HTML)** saves a self-contained report you can open in any browser or print.

### Model Comparison
1. Choose the **Scope**: all patients, or one patient.
2. Model cards show sensitivity, specificity, precision, F1, ROC-AUC, seizures detected, mean detection latency,
   false alarms per hour and inference time. Badges mark the best model on a metric.
3. The performance profile and scorecard compare all metrics side by side.
4. Tabs: confusion matrices, ROC and precision–recall curves, detection latency per seizure, the accuracy paradox,
   **Alarm tuning** (change the threshold and the K-of-N rule and see the clinical metrics update), CNN training curves.

### Explainability
1. Choose a held-out recording and a moment (or set the window end time with the slider).
2. The tiles show each model's seizure probability for that 4-second window and the annotation.
3. **Channel contribution**: head maps showing which electrodes drove each model's decision.
4. **CNN attention**: which samples of the window most influenced the network.
5. **What the random forest learned**: feature importance by family and the top 15 features.

## 4. Metrics in brief

| Metric | Meaning |
|---|---|
| Sensitivity (recall) | Share of seizure windows detected |
| Specificity | Share of non-seizure windows correctly left alone |
| Precision | Share of seizure predictions that were correct |
| F1-score | Balance of precision and sensitivity |
| ROC-AUC / PR-AUC | Ranking quality across all thresholds; PR-AUC is more informative for rare events |
| Seizures detected | Share of seizure events with at least one alarm |
| Detection latency | Seconds from annotated onset to the first alarm |
| False alarms per hour | Alarms outside any seizure, per hour of recording |

An alarm is raised when at least 3 of the last 4 one-second decisions are above 50 %.

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| "Trained models not found" | Run `python train.py` (about 7 minutes on a laptop CPU). |
| "No EDF files in data/raw" | Run `bash scripts/download_data.sh`. |
| `MemoryError` during training | Close other applications; training needs about 2 GB of free memory. |
| Theme looks mixed after switching | Reload the page once (Ctrl+R). |
| No sound in the Live Monitor | Click inside the monitor first; browsers block audio until you interact with the page. |
| Uploaded EDF is rejected | The file must contain the 18 bipolar channels of the CHB-MIT montage at a single sampling rate. |
