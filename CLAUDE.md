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
- Loads two pre-computed datasets at startup (cached via `@st.cache_data`): `explainability/alert_table/alert_table_6.parquet` and `processed_datasets/ueba_dataset_6/ueba_dataset_6_train.parquet`
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
  → CERT_Preprocessing.ipynb     # Feature engineering → ueba_dataset_5/ (train/test split, 108 features)
  → Autoencoder.ipynb            # Train autoencoder on insider-filtered normal data, extract 16-dim embeddings + scaler
  → Isolation_Forest.ipynb       # Train IF on normal-behavior embeddings, compute anomaly scores
  → Alert_Object_Builder.ipynb   # Merge scores + features → alert_table_5.parquet, cases_5.parquet
```

`prepare_data.py` provides shared utilities for the notebooks: `chronological_split` (90/10 train/test), `get_insiders`, `build_insider_mask`, and `get_scores`.

### Model Artifacts

- `encoders/encoder_model_6a/` — **recommended offline model**; trained on insider-filtered normal behavior from ueba_dataset_6; same architecture as model 5 (dropout 0.2, linear latent activation, early stopping patience=10, 80/10/10 chronological split, calibration-aware training); learning rate 0.001; training loss ~22.93%, validation loss ~15.81%
- `encoders/encoder_model_5/` — previous offline model; trained on ueba_dataset_5 (108 features); learning rate 0.001; training loss 26.78%, validation loss 16.90%
- `encoders/encoder_model_4/` — older offline model; identical architecture to model 5 but learning rate 0.0005; trained on ueba_dataset_4
- `isolation_forests/iforest_model_6a/` — **recommended offline IF**; trained on encoder_model_6a's normal-behavior embeddings; contamination="auto"; evaluation artifacts: confusion_matrix, precision_at_recall, user_detection_rate, alert_volume, time_to_first_alert, rank_order, score_distribution_shift
- `isolation_forests/iforest_model_5/` — previous offline IF; trained on model 5's normal-behavior embeddings; contamination=0.001; AUROC 0.765, separation ratio 1.27 (mean insider score 0.481 vs normal 0.378)
- `isolation_forests/iforest_model_4/` — older offline IF; trained on model 4's embeddings; contamination=0.001; AUROC 0.751, separation ratio 1.27
- Numbered suffixes (2, 3) are older experimental variants

### Risk Scoring

- Anomaly score → percentile rank against `anomaly_scores.npy` (calibration distribution — v6 uses an insider-free held-out calibration slice, not the training distribution)
- Four risk bands (assigned by `AlertObjectBuilder.assign_risk_band`):
  - CRITICAL: ≥ 95th percentile → `#ff1744` (bright red)
  - HIGH: ≥ 90th percentile → `#e84545` (red)
  - MEDIUM: ≥ 80th percentile → `#d4a017` (gold)
  - LOW: below 80th percentile → `#3a86a8` (steel blue)
- Both AE reconstruction error and IF anomaly score get independent risk bands (`ae_risk_band`, `if_risk_band`); dashboard primarily surfaces `ae_risk_band`
- v6 adds a `baseline_complete` gate: users with fewer than 14 days of history are not promoted to CRITICAL (prevents cold-start false positives)

### Key Design Patterns

- **Parquet-first I/O**: dashboard loads `.parquet` when available (5-10x faster than CSV)
- **Column downcast**: float64 → float32, int64 → int16/32 to reduce memory footprint
- **Reusable scripts**: `scripts/` contains class definitions used by both notebooks and runtime (Autoencoder, UEBAIsolationForest, Preprocessing, visualization helpers)

### Behavioral Features

v6 Layer A has 54 base features (up from 34 in v5); Layer B has 407 total columns (up from 108), expanded via per-user z-scores, multi-horizon rolling features, and peer-group z-scores.

**Layer A base channels:**
- **Auth** (3): logon_count, logoff_count, off_hours_logon
- **File** (6): file_open_count, file_write_count, file_copy_count, file_delete_count, unique_files_accessed, off_hours_files_accessed
- **Removable media** (3): usb_insert_count, usb_remove_count, off_hours_usb_usage
- **Email** (5): emails_sent, external_emails_sent, attachments_sent, off_hours_emails, unique_recipients
- **HTTP** (10): http_total_requests, http_visit_count, http_download_count, http_upload_count, http_jobsite_visits, http_cloud_storage_visits, http_suspicious_site_visits, off_hours_http_requests, http_long_url_count, unique_domains_visited
- **PC** (5): pcs_used_count, non_primary_pc_used_flag, non_primary_pc_http_requests_flag, non_primary_pc_usb_flag, non_primary_pc_file_copy_flag
- **Cross-channel flags** (7, derived in `apply_ueba_enhancements`): off_hours_activity_flag, usb_file_activity_flag, external_comm_activity_flag, jobsite_usb_activity_flag, suspicious_upload_flag, cloud_upload_flag, non_primary_pc_risk_flag

**v6 additions per channel (applied at Layer B):**
- **Sub-day intensity** (`<channel>_hourly_entropy`, `<channel>_peak_hour_count`, `<channel>_longest_active_run_minutes`): detects burst exfiltration and automated scripts invisible to daily totals
- **Late-night counters** (`<channel>_late_night_count`, 22:00–04:59): absolute signal preserved independently of per-user off-hours envelope
- **Multi-horizon rolling** (`<feature>_7d_sum`, `<feature>_30d_sum`, `<feature>_1d_over_30d_ratio`): surfaces accumulation patterns and single-day spikes relative to monthly baseline
- **Long-horizon z-scores** (`<feature>_zscore_90d`): 90-day trailing window to catch slow drift that outruns the 30-day z-score
- **Peer-group z-scores** (`<feature>_peer_zscore`): leave-one-out z-scores against LDAP role cohort; detects insiders whose own baseline has been corrupted

### Configuration Constants (in `scripts/Preprocessing.py`)

- `WORK_HOURS = (9, 17)` — population fallback for off-hours flags; v6 derives per-user envelopes from logon history (10th/90th percentile), persisted to `user_work_hours.parquet`
- `INTERNAL_EMAIL_DOMAIN = "dtaa.com"` — organization domain for external email detection
- Domain lists: `JOB_DOMAINS`, `CLOUD_STORAGE_DOMAINS`, `SUSPICIOUS_DOMAINS` for HTTP URL classification
- `PEER_GROUP_KEY` (in `config.py`) — LDAP grouping key for peer z-scores; default `role`, swappable to `department` or `team`

### Dataset Versions

- v1 (`ueba_dataset.csv`): 54 features
- v2 (`ueba_dataset_2.csv`): 78 features, adds PC-related signals
- v3b (`ueba_dataset_3b.csv/parquet`): 108 features, adds HTTP behavioral data
- v4 (`processed_datasets/ueba_dataset_4/`): 108 features, introduces chronological 90/10 train/test split at preprocessing time
- v5 (`processed_datasets/ueba_dataset_5/`): 108 features; resolves counting errors from previous iterations; 90/10 chronological split
  - `ueba_dataset_5a.csv` — (user, pc, day) level for drill-down
  - `ueba_dataset_5b.csv` — (user, day) level for model training
  - `ueba_dataset_5_train.csv/parquet` — first 90% chronologically (model training)
  - `ueba_dataset_5_test_stream.csv` — last 10% chronologically (live simulation / inference)
- v6 (`processed_datasets/ueba_dataset_6/`): 54 Layer A / 407 Layer B features — **currently active version**; adds per-user off-hours envelopes, bounded 30d/90d z-scores, peer-group z-scores, sub-day intensity features, late-night counters, multi-horizon rolling features, and an insider-free calibration split
  - `ueba_dataset_6a.parquet` — (user, pc, day) level for drill-down (54 features)
  - `ueba_dataset_6b.parquet` — (user, day) model-ready matrix (407 features)
  - `ueba_dataset_6_train.parquet` — first ~80% chronologically (model training)
  - `ueba_dataset_6_calibration.parquet` — middle ~10%, insiders removed (threshold calibration)
  - `ueba_dataset_6_calibration_eval.parquet` — middle ~10%, insiders retained (held-out evaluation)
  - `ueba_dataset_6_test_stream.parquet` — last ~10% chronologically (live simulation / inference)
  - `user_work_hours.parquet` — per-user `(start_hour, end_hour, schedule_complete)` table; must be reapplied at inference time
