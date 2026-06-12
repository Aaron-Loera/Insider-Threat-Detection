# Insider Threat Detection System

A UEBA (User and Entity Behavior Analytics) insider-threat detection platform built on the
CERT r6.2 synthetic dataset. An autoencoder learns compressed representations of daily user
behavior, an Isolation Forest scores those representations for anomalies, and a calibrated
alerting layer turns scores into SOC-ready, risk-banded alerts served through a Streamlit
dashboard — with a reproducible headless pipeline, CI, and tests behind it.

Built as a capstone project; production-*shaped* by design (see
[docs/CLEANUP_REPORT.md](docs/CLEANUP_REPORT.md) for what that means against a replayed
research dataset).

---

## Quickstart

```bash
# Dashboard runtime only (the exact pins Streamlit Cloud deploys)
pip install -r requirements.txt

# Training + pipeline + tests/lint (editable package with extras)
pip install -e .[ml,dev]
```

```bash
# Run the analyst dashboard
streamlit run dashboard/app.py

# Replay the held-out test stream through the live scorer (JSONL + WebSocket)
python live_simulation.py --interval 0.5

# Audit / run the ML pipeline (requires the local CERT data tree)
python -m ueba.pipeline status
python -m ueba.pipeline all

# Tests and lint
pytest
ruff check .
```

---

## Repository layout

```
├── src/ueba/                 # installable package (pip install -e .)
│   ├── config.py             #   central path/version registry (paths.local.py overrides)
│   ├── constants.py          #   feature-engineering constants (work hours, domain lists)
│   ├── risk.py               #   single source of truth for risk bands + percentiles
│   ├── features/             #   CERT preprocessing, Layer A/B engineering, work hours
│   ├── models/               #   autoencoder, isolation forest, data preparation
│   ├── alerts/               #   alert object builder, reconstruction-error explainer
│   ├── serving/              #   live scoring, dashboard dataset build/upload
│   ├── pipeline/             #   headless stage CLI + fail-fast artifact manifest
│   └── viz/                  #   analysis/visualization helpers
├── dashboard/                # Streamlit app (Cloud entrypoint) + SQLite triage store
├── tests/                    # pytest suite — synthetic fixtures, no real data needed
├── docs/                     # PIPELINE.md (stage contracts), CLEANUP_REPORT.md (roadmap)
├── *.ipynb                   # training notebooks — narrative documentation; the
│                             #   pipeline is the production path
├── scripts/, config.py, ...  # back-compat shims re-exporting from ueba.*
├── processed_datasets/       # v6 datasets + splits (gitignored, built by the pipeline)
├── encoders/, isolation_forests/, explainability/   # v6 model + alert artifacts
└── legacy/                   # archived v1–v5 artifact generations (gitignored)
```

---

## How it works

```
raw CERT logs (logon/file/device/email/http)
  → preprocess      54 Layer A channels → 414 Layer B features per (user, day):
  │                 per-user z-scores, multi-horizon rolling, peer-group z-scores,
  │                 sub-day intensity, per-user work-hour envelopes
  → train-ae        autoencoder on insider-filtered normal behavior → 16-dim embeddings
  → train-if        isolation forest on normal-behavior embeddings → anomaly scores
  → calibrate       absolute risk thresholds from an insider-free calibration slice
  → build-alerts    risk-banded alerts (LOW/MEDIUM/HIGH/CRITICAL) + multi-day cases
  → dashboard       overview KPIs, per-user investigation, alert triage, channel analysis
```

Risk bands come from one shared module (`ueba.risk`) used identically by the offline
alert builder and the live scorer. Users with fewer than 14 days of history are never
promoted to CRITICAL (cold-start gate). Every pipeline stage validates its input
artifacts through a fail-fast manifest — a missing artifact names the stage that
produces it. See [docs/PIPELINE.md](docs/PIPELINE.md).

---

## Setup notes

- **CERT data:** preprocessing needs the raw CERT r6.2 CSVs locally. Copy
  `paths.local.example.py` to `paths.local.py` and set `CERT_PATH`. All other paths
  derive from `MODEL_VERSION` and need no configuration.
- **Streamlit Cloud:** the dashboard deploys from `requirements.txt` (hand-pinned
  lockfile — edit deliberately) + `runtime.txt`, loading a slim pre-merged serving
  dataset from the Hugging Face Hub. Build it with
  `python -m ueba.pipeline build-dashboard` and publish with
  `python -m ueba.serving.upload_dashboard_dataset`.
- **WebSocket:** the live simulator broadcasts on `ws://localhost:8765`,
  unauthenticated by design and bound to localhost only.

---

## Limitations

- CERT synthetic dataset replayed offline — no live SIEM/streaming integration
- Alerts are probabilistic and require analyst review
- No automated incident response
- Built for academic and demonstration purposes

---

## Team

* Aaron Lorea - https://www.linkedin.com/in/aaronloera324/
* Tyler Kees - https://www.linkedin.com/in/tyler-kees/
* Melusi Senzanje - https://www.linkedin.com/in/melusisenzanje/
* Melody Nnadi - https://www.linkedin.com/in/melodynnadi/
* Hugo Margues - https://www.linkedin.com/in/hugomarquesnob/
* Matthew Emanuel - https://www.linkedin.com/in/matthew-emanuel-1b168a340/

---

## License

This project is developed for academic purposes. Use of third-party libraries must
comply with their respective licenses.

## Acknowledgments

* CERT Insider Threat Dataset (Carnegie Mellon University)
* TensorFlow, Scikit-learn, and Streamlit communities
