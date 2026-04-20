# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the dashboard:**
```bash
streamlit run dashboard/app.py
```

**Run live scoring/simulation:**
```bash
python live_simulation.py
python live_simulation.py --interval 0.5 --input processed_datasets/test_stream.csv --output processed_datasets/live_results.jsonl --port 8765
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

Note: `requirements.txt` only lists dashboard dependencies. ML pipeline notebooks also require `tensorflow`, `scikit-learn`, `joblib`, `websockets`, and `scipy`.

## Architecture

This is a UEBA (User and Entity Behavior Analytics) insider threat detection system built on the CERT dataset.

### Two Runtime Components

**1. `dashboard/app.py`** — Streamlit web UI for security analysts
- Loads two pre-computed datasets at startup (cached via `@st.cache_data`): `explainability/alert_table/alert_table_5.parquet` and `processed_datasets/ueba_dataset_5/ueba_dataset_5_train.parquet`
- Merges them on `(user, day)` keys
- Four tabs: Overview (KPIs + charts), Investigation (per-user deep dive), Alerts (filterable feed), Channels (feature analysis)
- Global sidebar filters (date range, risk level, user search) drive all views

**2. `live_simulation.py`** — Real-time scoring engine
- `LiveScorer` class loads encoder model, StandardScaler, and Isolation Forest once at startup
- Streams `test_stream.csv` row-by-row through the ML pipeline: scale → embed (16-dim latent) → IF score → percentile rank → risk level
- Outputs scored records to `processed_datasets/live_results.jsonl` (one JSON object per line)
- Broadcasts each scored record via WebSocket (default port 8765) for real-time dashboard updates
- Tracks `_score_ms` per record for latency diagnostics

### ML Pipeline (Offline, in Jupyter notebooks)

```
Raw CERT logs (logon/file/device/email/http CSVs)
  → CERT_Preprocessing.ipynb     # Feature engineering → ueba_dataset_4/ (train/test split, 108 features)
  → Autoencoder.ipynb            # Train autoencoder on insider-filtered normal data, extract 16-dim embeddings + scaler
  → Isolation_Forest.ipynb       # Train IF on normal-behavior embeddings, compute anomaly scores
  → Alert_Object_Builder.ipynb   # Merge scores + features → alert_table_5.parquet, cases_5.parquet
```

`prepare_data.py` provides shared utilities for the notebooks: `chronological_split` (90/10 train/test), `get_insiders`, `build_insider_mask`, and `get_scores`.

### Model Artifacts

- `encoders/encoder_model_1/` — used by `live_simulation.py` (trained on ueba_dataset v1, 54 features)
- `encoders/encoder_model_5/` — **recommended offline model**; trained on insider-filtered normal behavior from ueba_dataset_5; same architecture as model 4 (dropout 0.2, linear latent activation, early stopping patience=10, 90/10 chronological split, 85/15 train/val split on normal data); learning rate 0.001; training loss 26.66%, validation loss 17.04%
- `encoders/encoder_model_4/` — previous offline model; identical architecture to model 5 but learning rate 0.0005; trained on ueba_dataset_4
- `isolation_forests/iforest_model_1/` — used by `live_simulation.py` (contamination=0.05)
- `isolation_forests/iforest_model_5/` — **recommended offline IF**; trained on model 5's normal-behavior embeddings; contamination=0.001; AUROC 0.750, separation ratio 1.25 (mean insider score 0.471 vs normal 0.377)
- `isolation_forests/iforest_model_4/` — previous offline IF; trained on model 4's embeddings; contamination=0.001; AUROC 0.751, separation ratio 1.27
- Numbered suffixes (2, 3) are older experimental variants

### Risk Scoring

- Anomaly score → percentile rank against `anomaly_scores.npy` (training distribution)
- Four risk bands (assigned by `AlertObjectBuilder.assign_risk_band`):
  - CRITICAL: ≥ 95th percentile → `#ff1744` (bright red)
  - HIGH: ≥ 90th percentile → `#e84545` (red)
  - MEDIUM: ≥ 80th percentile → `#d4a017` (gold)
  - LOW: below 80th percentile → `#3a86a8` (steel blue)
- Both AE reconstruction error and IF anomaly score get independent risk bands (`ae_risk_band`, `if_risk_band`); dashboard primarily surfaces `ae_risk_band`

### Key Design Patterns

- **Parquet-first I/O**: dashboard loads `.parquet` when available (5-10x faster than CSV)
- **Column downcast**: float64 → float32, int64 → int16/32 to reduce memory footprint
- **Reusable scripts**: `scripts/` contains class definitions used by both notebooks and runtime (Autoencoder, UEBAIsolationForest, Preprocessing, visualization helpers)

### Behavioral Features

17 raw features across 4 channels per user-day:
- **Auth**: logon_count, logoff_count, off_hours_logon
- **File**: file_open/write/copy/delete counts, unique_files_accessed, off_hours_files_accessed
- **Removable media**: usb_insert/remove counts, off_hours_usb_usage
- **Email**: emails_sent, unique_recipients, external_emails, attachments_sent, off_hours_emails
- **Cross-channel flags**: USB+File co-occurrence, off-hours activity, external communication

### Configuration Constants (in `scripts/Preprocessing.py`)

- `WORK_HOURS = (9, 17)` — defines business hours for off-hours flags
- `INTERNAL_EMAIL_DOMAIN = "dtaa.com"` — organization domain for external email detection
- Domain lists: `JOB_DOMAINS`, `CLOUD_STORAGE_DOMAINS`, `SUSPICIOUS_DOMAINS` for HTTP URL classification

### Dataset Versions

- v1 (`ueba_dataset.csv`): 54 features
- v2 (`ueba_dataset_2.csv`): 78 features, adds PC-related signals
- v3b (`ueba_dataset_3b.csv/parquet`): 108 features, adds HTTP behavioral data
- v4 (`processed_datasets/ueba_dataset_4/`): 108 features; introduces chronological 90/10 train/test split at preprocessing time
- v5 (`processed_datasets/ueba_dataset_5/`): 108 features — **currently active version**; fixes unique counting bugs from prior versions, adds z-score clipping for improved model performance, pipeline optimizations; same split structure as v4:
  - `ueba_dataset_5a.csv/parquet` — (user, pc, day) level for drill-down
  - `ueba_dataset_5b.csv/parquet` — (user, day) level for model training
  - `ueba_dataset_5_train.csv/parquet` — first 90% chronologically (model training)
  - `ueba_dataset_5_test_stream.csv/parquet` — last 10% chronologically (live simulation / inference)
