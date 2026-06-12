# Pipeline Reference — `python -m ueba.pipeline`

The headless, reproducible counterpart to the training notebooks. Stages call
the same `ueba.*` package functions the notebooks orchestrate; the notebooks
remain the narrative documentation, the pipeline is the production path.

```
raw CERT logs
  └─ preprocess        ueba_dataset_{V}{a,b}.parquet, train/calibration/
  │                    calibration_eval/test_stream splits, user_work_hours,
  │                    peer_baselines_{V}.parquet
  └─ train-ae          autoencoder/encoder .keras, feature_scaler.pkl,
  │                    feature_cols.json (train/serve contract), embeddings,
  │                    ae_baseline_clean.npy, metrics_ae.json
  └─ train-if          iforest_model.pkl, anomaly_scores.npy,
  │                    if_baseline_clean.npy, metrics_if.json
  └─ explain           reconstruction_error_table_{V}[_test].parquet
  │                    (--split train | test_stream)
  └─ calibrate         calibration_thresholds.json (band-keyed: ae/if x
  │                    LOW/MEDIUM/HIGH/CRITICAL) + clean baselines
  └─ build-alerts      alert_table_{V}[_calib|_test].parquet + cases
  │                    (--split main | calib | test)
  └─ build-dashboard   ueba_dataset_{V}_dashboard.parquet (slim serving layer)
```

## Commands

```bash
python -m ueba.pipeline status                      # audit the artifact tree
python -m ueba.pipeline all                         # full run, stops at first failure
python -m ueba.pipeline preprocess
python -m ueba.pipeline train-ae [--epochs 100]
python -m ueba.pipeline train-if
python -m ueba.pipeline explain --split train
python -m ueba.pipeline explain --split test_stream
python -m ueba.pipeline calibrate [--thresholds-only]
python -m ueba.pipeline build-alerts --split main
python -m ueba.pipeline build-alerts --split test
python -m ueba.pipeline build-alerts --split calib
python -m ueba.pipeline build-dashboard
```

`ueba-pipeline` (console script installed with the package) is equivalent.
Requires the ML extras: `pip install -e .[ml]`.

## Stage contracts and the manifest

Every stage declares `requires()` (input artifacts + the stage that produces
each) and `produces()`. Before running, inputs are validated through
`ueba.pipeline.manifest.require()` — a missing artifact is a **hard error
naming the producing stage**, never a silent skip. This replaced the historical
behavior where `Alert_Object_Builder` quietly skipped the test alert table
when the test reconstruction table was absent (CLEANUP_REPORT gap 3).
`build-alerts --allow-missing` is the explicit escape hatch.

After running, stages journal what they produced (size, content hash prefix,
git commit, model version, timestamp) to
`processed_datasets/ueba_dataset_{V}/pipeline_manifest.json`.
`status` re-validates both the expected artifact tree and the journal.

## Versioning

All paths derive from `MODEL_VERSION` in `ueba.config`. Precedence:

1. `--version N` on the CLI (sets `UEBA_MODEL_VERSION` before config loads)
2. `MODEL_VERSION` in `paths.local.py`
3. default (`"6"`)

`CERT_VERSION` (default `"6.2"`) selects the insider ground-truth rows from
`answers/insiders.csv` and is overridable in `paths.local.py`.

## Calibration thresholds — format note

`calibration_thresholds.json` is written **only** by the `calibrate` stage, in
the band-keyed format both consumers read:

```json
{"ae": {"LOW": 312.1, "MEDIUM": 444.9, "HIGH": 507.2, "CRITICAL": null},
 "if": {"LOW": 0.4631, "MEDIUM": 0.5072, "HIGH": 0.5239, "CRITICAL": null}}
```

(`CRITICAL: null` = no upper bound.) Historically `Isolation_Forest.ipynb`
wrote a percentile-keyed file to the same path which
`Alert_Object_Builder.ipynb` then overwrote with this format; the pipeline
removes that conflict. `calibrate --thresholds-only` derives the JSON from the
existing clean-baseline `.npy` files without rescoring — the fast recovery
path when only the JSON is missing.

## Evaluation metrics

`train-ae` and `train-if` persist their evaluation suites as machine-readable
JSON next to the PNG plots (`encoders/encoder_model_{V}/metrics_ae.json`,
`isolation_forests/iforest_model_{V}/metrics_if.json`): AUROC, average
precision, recall at the 80/90/95th baseline percentiles, and for the IF the
SOC battery (confusion matrix, precision@recall, user detection rate, alert
volume, time-to-first-alert, score-distribution shift). Resubstitution
numbers are labeled as diagnostics; the calibration block is the
authoritative reference.

## Inference-time work hours

`user_work_hours.parquet` (per-user envelopes, derived in preprocess) is the
table CLAUDE.md says must be reapplied at inference. `ueba.features.work_hours`
provides `apply_off_hours_flags()` (event-level flagging with population
fallback — usable by any raw-event ingestion) and the live scorer now loads
the table and **warns once per cold-start user** whose off-hours features were
necessarily built with the population default. Scores on the pre-featurized
test stream are unchanged. Full raw-event scoring (running Layer A/B at
inference) remains future work and is out of scope for the CERT replay.

## What CI runs

Unit tests with synthetic fixtures and stub models only. The pipeline itself
needs the local CERT data tree (10+ GB) and is a local-only flow;
`python -m ueba.pipeline status` is the manual gate before publishing
artifacts.
