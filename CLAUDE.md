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
python live_simulation.py --interval 0.5 --input processed_datasets/ueba_dataset_6/ueba_dataset_6_test_stream.parquet --output processed_datasets/live_results.jsonl --port 8765
```

**Run the ML pipeline (headless):**
```bash
python -m ueba.pipeline status          # audit the artifact tree
python -m ueba.pipeline all             # full run (needs local CERT data)
python -m ueba.pipeline <stage>         # preprocess | train-ae | train-if | explain | calibrate | build-alerts | build-dashboard
```
See `docs/PIPELINE.md` for stage contracts. The training notebooks remain as documentation; the pipeline is the production path.

**Run tests / lint:**
```bash
pytest
ruff check .
```

**Install dependencies:**
```bash
pip install -r requirements.txt    # dashboard runtime (Streamlit Cloud lockfile)
pip install -e .[ml,dev]           # training + pipeline + test/lint stack
```

## Architecture

This is a UEBA (User and Entity Behavior Analytics) insider threat detection system built on the CERT dataset.

### Two Runtime Components

**1. `dashboard/app.py`** — Streamlit web UI for security analysts
- Loads a pre-merged slim serving dataset at startup (cached): `ueba_dataset_6_dashboard.parquet`, read locally when present or downloaded from the Hugging Face dataset repo (`HF_DATASET_REPO` in `config.py`) via `ueba.serving.hf_io.get_dataset_file` — the same local-cache-first helper backs the Investigation tab's per-user explanation fallback (`details/{user}.parquet`) and `live_replay.py`'s cloud fallback (`alert_table_6.parquet`)
- The serving dataset is built offline by `scripts/build_dashboard_dataset.py` (merges the alert table with training features on `(user, day)`, projects columns, downcasts dtypes) and published — along with the rest of the version's manifest (dataset splits, alert table, model ensemble) — by `scripts/upload_to_hf.py`
- Re-enforces the `baseline_complete` gate at load time: CRITICAL bands are demoted to HIGH for any user with fewer than 14 days of history, guarding against alert tables regenerated without the filter
- Four tabs: Overview (KPIs + charts), Investigation (per-user deep dive), Alerts (filterable feed), Channels (feature analysis)
- Global sidebar filters (date range, risk level, user search) drive all views

**`dashboard/db.py`** — SQLite-backed alert disposition store
- `init_db()` creates the `alert_dispositions` table on first run (`alert_state.db` lives alongside `app.py`)
- `upsert_disposition(user, day, status, note)` — idempotent upsert keyed on `(user, day)`
- `get_disposition` / `get_all_dispositions` — read-path helpers for the dashboard

**2. `live_simulation.py`** — Real-time scoring engine
- `LiveScorer` class loads encoder model, StandardScaler, and Isolation Forest once at startup
- Streams the v6 test_stream parquet row-by-row through the ML pipeline: scale → embed (16-dim latent) → IF score → percentile rank → risk level
- Outputs scored records to `processed_datasets/live_results.jsonl` (one JSON object per line)
- Broadcasts each scored record via WebSocket (default port 8765) for real-time dashboard updates
- **Security note:** the WebSocket server is unauthenticated by design and binds to `localhost` only (`live_simulation.py`, `websockets.serve(..., "localhost", port)`). It must not be exposed beyond the local machine without adding authentication.
- Tracks `_score_ms` per record for latency diagnostics

### ML Pipeline (Offline)

The production path is the headless CLI (`python -m ueba.pipeline <stage>`, see `docs/PIPELINE.md`):

```
Raw CERT logs (logon/file/device/email/http CSVs)
  → preprocess      # ueba_dataset_6/ (train/calibration/test_stream splits, 54 Layer A → 414 Layer B features, work-hour envelopes, peer baselines)
  → train-ae        # autoencoder on insider-filtered normal data → 16-dim embeddings + scaler + feature_cols.json contract + clean AE baseline
  → train-if        # IF on normal-behavior embeddings → anomaly scores + clean IF baseline; metrics_if.json
  → explain         # reconstruction-error tables (--split train | test_stream)
  → calibrate       # band-keyed calibration_thresholds.json from the insider-free baseline (sole producer)
  → build-alerts    # alert_table_6[_calib|_test].parquet + cases (--split main | calib | test); inner-join asserts surface pipeline drift early
  → build-dashboard # slim serving parquet
```

Every stage validates inputs through a fail-fast manifest (`ueba.pipeline.manifest`) — a missing artifact names its producing stage. `python -m ueba.pipeline status` audits the artifact tree. The five training notebooks (CERT_Preprocessing, Autoencoder, Isolation_Forest, Reconstruction_Error_Explainer, Alert_Object_Builder) remain as narrative documentation of the same flow.

`ueba.models.data_prep` (shimmed at root as `prepare_data.py`) provides shared utilities: `chronological_split`, `get_insiders`, `build_insider_mask`, `to_model_matrix`, and `get_scores`.

### Model Artifacts

- `encoders/encoder_model_6/` — **active model**; trained on insider-filtered normal behavior from ueba_dataset_6 (dropout 0.2, linear latent activation, early stopping patience=10, 80/10/10 chronological split, calibration-aware training); learning rate 0.001; training loss ~22.93%, validation loss ~15.81%. Also holds `feature_scaler.pkl`, `feature_cols.json` (train/serve column contract), `ae_baseline_clean.npy`, `calibration_thresholds.json`, and `metrics_ae.json`
- `isolation_forests/iforest_model_6/` — **active IF**; trained on encoder_model_6's normal-behavior embeddings; contamination="auto"; evaluation artifacts as PNGs + machine-readable `metrics_if.json` (confusion_matrix, precision_at_recall, user_detection_rate, alert_volume, time_to_first_alert, rank_order, score_distribution_shift)
- Generations v1–v5 (encoders, isolation forests, datasets, alert/reconstruction tables) are archived under gitignored `legacy/`, mirroring their original directory structure; version-history notes live in `encoders/encoder_details.txt`, `isolation_forests/iforest_details.txt`, and `processed_datasets/dataset_notes.txt`

### Risk Scoring

- **`ueba.risk` is the single source of truth** for band assignment and percentile ranking — used identically by the offline `AlertObjectBuilder` and the live scorer (scalar + vectorized forms; a percentile equal to a threshold belongs to the higher band)
- Anomaly score → percentile rank against the clean calibration baseline (insider-free held-out slice, not the training distribution); when `calibration_thresholds.json` exists (produced by `ueba.pipeline calibrate`), banding uses calibrated **absolute** thresholds targeting a clean-day alert budget (~5% LOW / 1% MEDIUM / 0.5% HIGH ceilings) instead of percentile cutoffs
- Four risk bands (percentile fallback semantics):
  - CRITICAL: ≥ 95th percentile → `#ff1744` (bright red)
  - HIGH: ≥ 90th percentile → `#e84545` (red)
  - MEDIUM: ≥ 80th percentile → `#d4a017` (gold)
  - LOW: below 80th percentile → `#3a86a8` (steel blue)
- Both AE reconstruction error and IF anomaly score get independent risk bands (`ae_risk_band`, `if_risk_band`); dashboard primarily surfaces `ae_risk_band`
- v6 adds a `baseline_complete` gate: users with fewer than 14 days of history are not promoted to CRITICAL (prevents cold-start false positives)

### Key Design Patterns

- **Parquet-first I/O**: dashboard loads `.parquet` when available (5-10x faster than CSV)
- **Column downcast**: float64 → float32, int64 → int16/32 to reduce memory footprint
- **Installable package**: all shared logic lives in `src/ueba/` (`pip install -e .`); the root modules (`config.py`, `prepare_data.py`, `live_simulation.py`) and everything under `scripts/` are back-compat shims that re-export from `ueba.*` with a source-tree fallback — the dashboard and notebooks work whether or not the package is installed
- **Layer A safepoint**: `save_nunique_frames(nunique_frames, safepoint_dir)` / `load_nunique_frames(safepoint_dir)` in `ueba.features.preprocessing` persist the intermediate nunique identity frames to parquet so `build_layer_b()` can resume after a kernel restart without rerunning Layer A

### Behavioral Features

v6 Layer A has 54 base features (up from 34 in v5); Layer B has 414 total columns (up from 108), expanded via per-user z-scores, multi-horizon rolling features, peer-group z-scores, and user profile enrichment.

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
- **User profile enrichment** (joined by `build_layer_b`): `employee_name`, `department`, `role`, `supervisor`, `functional_unit`, `is_active`, `role_sensitivity` (0–1 float32; computed by `compute_role_sensitivity()` — executives/finance/IT-admin: 0.8–1.0, standard employees: 0.3–0.5)

### Configuration Constants (in `src/ueba/constants.py`)

- `WORK_HOURS = (9, 17)` — population fallback for off-hours flags; v6 derives per-user envelopes from logon history (10th/90th percentile), persisted to `user_work_hours.parquet` and reapplied at inference via `ueba.features.work_hours` (the live scorer warns once per cold-start user)
- `INTERNAL_EMAIL_DOMAIN = "dtaa.com"` — organization domain for external email detection
- Domain lists: `JOB_DOMAINS`, `CLOUD_STORAGE_DOMAINS`, `SUSPICIOUS_DOMAINS` for HTTP URL classification
- `PEER_GROUP_KEY` (in `config.py`) — LDAP grouping key for peer z-scores; default `role`, swappable to `department` or `team`
- `PEER_BASELINES_PATH` (in `config.py`) — path to `peer_baselines_{V}.parquet` (department × day × feature means); generated from training data; dashboard degrades gracefully when absent

### Dataset Versions

Versions v1–v5 are archived under gitignored `legacy/` (e.g. `legacy/processed_datasets/ueba_dataset_5/`); only v6 remains in the active tree.

- v1 (`ueba_dataset.csv`): 54 features
- v2 (`ueba_dataset_2.csv`): 78 features, adds PC-related signals
- v3b (`ueba_dataset_3b.csv/parquet`): 108 features, adds HTTP behavioral data
- v4 (`processed_datasets/ueba_dataset_4/`): 108 features, introduces chronological 90/10 train/test split at preprocessing time
- v5 (`processed_datasets/ueba_dataset_5/`): 108 features; resolves counting errors from previous iterations; 90/10 chronological split
  - `ueba_dataset_5a.csv` — (user, pc, day) level for drill-down
  - `ueba_dataset_5b.csv` — (user, day) level for model training
  - `ueba_dataset_5_train.csv/parquet` — first 90% chronologically (model training)
  - `ueba_dataset_5_test_stream.csv` — last 10% chronologically (live simulation / inference)
- v6 (`processed_datasets/ueba_dataset_6/`): 54 Layer A / 414 Layer B features — **currently active version**; adds per-user off-hours envelopes, bounded 30d/90d z-scores, peer-group z-scores, sub-day intensity features, late-night counters, multi-horizon rolling features, user profile enrichment (role_sensitivity, is_active, etc.), and an insider-free calibration split
  - `ueba_dataset_6a.parquet` — (user, pc, day) level for drill-down (54 features)
  - `ueba_dataset_6b.parquet` — (user, day) model-ready matrix (407 features)
  - `ueba_dataset_6_train.parquet` — first ~80% chronologically (model training)
  - `ueba_dataset_6_calibration.parquet` — middle ~10%, insiders removed (threshold calibration)
  - `ueba_dataset_6_calibration_eval.parquet` — middle ~10%, insiders retained (held-out evaluation)
  - `ueba_dataset_6_test_stream.parquet` — last ~10% chronologically (live simulation / inference)
  - `user_work_hours.parquet` — per-user `(start_hour, end_hour, schedule_complete)` table; must be reapplied at inference time
  - `peer_baselines_6.parquet` — (department, day, \<feature\> means) table used by the Investigation tab for peer-comparison charts
