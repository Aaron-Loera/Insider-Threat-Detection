# Repository Cleanup Report — Production-Readiness Roadmap

**Date:** 2026-06-11
**Scope:** Full repository audit + phased transformation plan
**Goal:** Remodel this research-grade UEBA project into a production-shaped insider-threat
detection workflow — CI/CD, reproducible headless pipeline, packaging, tests, and a clean
repository structure — so future work can iterate toward a true SOC detection system.

---

## 1. Framing: what "production-level" means here

This system is built on the **CERT synthetic insider-threat dataset**, replayed offline.
There is no live SIEM feed, no streaming log source, and no real-time enrichment service.
That constrains what "production" can mean:

| In scope (production-*shaped*) | Out of scope (needs real infra) |
|---|---|
| Reproducible, headless ML pipeline (CLI, not notebooks) | Real streaming ingestion (Kafka/SIEM connectors) |
| CI: lint + unit tests on every PR | Running the actual pipeline in CI (needs 10+ GB data) |
| Installable package with declared dependencies | Scheduled retraining / model registry services |
| Single source of truth for scoring logic | Authentication hardening of the local WebSocket |
| Artifact manifests + fail-fast validation between stages | Multi-tenant / horizontally-scaled serving |
| Versioned model artifacts with machine-readable metrics | Online feature stores |

The live-simulation component (`live_simulation.py`) remains the stand-in for a streaming
source: it replays the chronologically held-out `test_stream` slice through the scoring
path, which is the correct pattern given the dataset.

---

## 2. Current-state inventory (audited 2026-06-11)

- **Disk footprint:** ~36 GB total; only **54 files tracked in git** (code, notebooks, configs).
- **Active generation:** v6 — `processed_datasets/ueba_dataset_6/`, `encoders/encoder_model_6/`,
  `isolation_forests/iforest_model_6/`, `explainability/alert_table/alert_table_6/`,
  `explainability/reconstruction_error/reconstruction_error_table_6.parquet`.
- **Stale generations v1–v5 (~32 GB, all gitignored, local-disk only):**
  - Datasets v0–v5: ~8 GB (`processed_datasets/ueba_dataset{,_2,_3,_4,_5}/`)
  - Encoder models 1–5: ~580 MB; Isolation Forests 1–5: ~84 MB
  - Old alert tables (v1–v5): ~1.1 GB; old reconstruction-error tables (v1–v5 CSVs): ~21 GB
- **Runtime components:**
  - `dashboard/app.py` — 4,309-line monolithic Streamlit app; deploys to **Streamlit Cloud**
    (entrypoint path, root `requirements.txt` + `runtime.txt` are deployment contracts).
  - `live_simulation.py` — `LiveScorer` + WebSocket broadcast (port 8765) + JSONL output.
  - `dashboard/db.py` — SQLite alert-disposition store.
- **ML pipeline:** 4 orchestration notebooks (`CERT_Preprocessing` → `Autoencoder` →
  `Isolation_Forest` → `Alert_Object_Builder`) + `Reconstruction_Error_Explainer` on a
  parallel track. Nearly all business logic already lives in `scripts/` classes — the
  notebooks are orchestration + documentation. Good foundation for a CLI extraction.
- **Quality infrastructure:** none. No tests, no CI, no `pyproject.toml`, no lint config,
  no pre-commit.
- **Dependencies:** `requirements.txt` covers the dashboard only (pinned for Streamlit
  Cloud). ML deps (tensorflow, scikit-learn, joblib, websockets, scipy) are undeclared.

---

## 3. Audit findings

Nine gaps, ordered by severity. File:line references are as of commit `8487770`.

### Gap 1 — Risk-band logic duplicated with conflicting boundary semantics *(correctness)*
Two independent implementations disagree at exact percentile boundaries:
- `scripts/AlertObjectBuilder.py:112–126` (`assign_risk_band`) uses `percentile <= thresh`,
  so **p = 95.0 → HIGH** and p = 80.0 → LOW.
- `live_simulation.py:46–48` (`_FALLBACK_*` constants) + inline banding (~lines 168–181)
  use `>=`, so **p = 95.0 → CRITICAL** and p = 80.0 → MEDIUM.
CLAUDE.md documents the `>=` semantics ("CRITICAL: ≥ 95th percentile"). If thresholds ever
change, two places must be edited in lockstep. **Fix (Phase 3):** one shared module,
canonical `>=` semantics; the offline and live paths both import it.

### Gap 2 — `user_work_hours.parquet` never reapplied at inference *(correctness)*
v6 derives per-user off-hours envelopes during preprocessing and persists them, and
CLAUDE.md states they "must be reapplied at inference time" — but no inference code does
this. It works today only because `test_stream` is pre-featurized; any *truly new* user
streamed through `LiveScorer` would get population-default off-hours flags silently.
**Fix (Phase 5):** `apply_off_hours_flags()` utility wired into `LiveScorer` behind a
`--raw-input` flag (default off → current behavior byte-identical).

### Gap 3 — Missing test-period alert table; silent skip *(correctness / observability)*
`alert_table_6_test.parquet` was never generated because `Reconstruction_Error_Explainer`
was never run on `test_stream`. `Alert_Object_Builder.ipynb` detects the missing input and
**silently skips** the test alert table instead of failing. **Fix (Phase 5):** pipeline
manifest converts missing-artifact conditions into hard errors naming the producing stage;
generate the missing artifact.

### Gap 4 — Bare `except Exception: pass` swallowing errors *(operability)*
- `dashboard/app.py` ~line 1754 (live-results parsing)
- `live_replay.py:103, 116`
- `scripts/build_dashboard_dataset.py:124` (`except Exception` around `import config`)
Failures disappear without a trace. **Fix (Phase 2):** narrow exception types where the
intent is clear; where the swallow is deliberate, keep it but log via `logging.warning`.

### Gap 5 — Preprocessing constants hardcoded outside config *(configurability)*
`scripts/Preprocessing.py:13–58`: `WORK_HOURS`, `INTERNAL_EMAIL_DOMAIN="dtaa.com"`,
`LONG_URL_THRESHOLD=90`, `JOB_DOMAINS`, `CLOUD_STORAGE_DOMAINS`, `SUSPICIOUS_DOMAINS`,
dtype/usecols maps. `config.py` is otherwise a good centralized registry; these bypass it.
**Fix (Phase 4):** consolidated into `src/ueba/constants.py`.

### Gap 6 — Evaluation metrics not machine-readable *(MLOps)*
AUROC, precision@recall, user-detection-rate, time-to-first-alert, etc. are computed at
training time but persisted only as PNG plots. No JSON/CSV metrics live beside the model
artifacts, so nothing can diff model quality across versions programmatically.
**Fix (Phase 5):** `compute_*` methods also write `metrics_ae.json` / `metrics_if.json`
into the model artifact directories.

### Gap 7 — Monolithic, untestable dashboard *(maintainability — deferred)*
`dashboard/app.py` is 4,309 lines with module-level auth + data loading and magic numbers
(`MAX_PLOT_POINTS = 50_000` ~line 1879; `_MAX_LIVE_ROWS = 5_000` ~line 2714). Decomposition
into `dashboard/lib/` + Streamlit `pages/` is **deferred to its own PR series (Phase 7)** —
it is the highest-churn, highest-regression-risk item and needs its own staging runs.

### Gap 8 — Junk tracked in git *(hygiene)*
Verified in the index: `scripts/__pycache__/IsolationForestScoreDistribution.cpython-312.pyc`,
`dashboard/alert_state.db` (runtime SQLite, gitignore rule added after it was tracked),
`Dashboard.html` (440-byte stub referenced nowhere). **Fixed in Phase 1.**

### Gap 9 — Undocumented WebSocket trust model *(security documentation)*
`live_simulation.py` serves WebSocket on `localhost:8765` with no authentication. This is
acceptable **only** because it binds to localhost; that assumption was undocumented.
**Fixed in Phase 1** (documented in CLAUDE.md). Adding auth is explicitly out of scope.

---

## 4. Target repository layout

Package name **`ueba`** (matches existing vocabulary: `UEBA_PATH`, `UEBAIsolationForest`).

```
├── pyproject.toml                  # packaging + ruff + pytest config
├── requirements.txt                # hand-pinned Streamlit Cloud lockfile (+ "-e ." in Phase 4)
├── runtime.txt                     # 3.12 (Streamlit Cloud)
├── config.py / prepare_data.py / live_simulation.py / live_replay.py
│                                   # thin back-compat shims re-exporting from ueba.*
├── src/ueba/
│   ├── config.py                   # BASE_DIR: env UEBA_BASE_DIR or marker-walk to repo root
│   ├── constants.py                # WORK_HOURS, domains, dtype maps          [gap 5]
│   ├── risk.py                     # single source of truth for risk bands    [gap 1]
│   ├── features/                   # layer_a, layer_b, splits, work_hours (split of Preprocessing.py)
│   ├── models/                     # autoencoder, isolation_forest, evaluation (metrics→JSON), data_prep
│   ├── alerts/                     # builder (AlertObjectBuilder), explainer
│   ├── serving/                    # live_scorer, live_simulation, live_replay,
│   │                               # build_dashboard_dataset, hf_upload
│   ├── pipeline/                   # cli, manifest, stages/{preprocess,train_ae,train_if,
│   │                               #   explain,build_alerts,build_dashboard}
│   └── viz/                        # 3 visualization helpers
├── dashboard/app.py + db.py        # UNCHANGED paths (Streamlit Cloud entrypoint)
├── scripts/                        # back-compat shims for notebooks; csv_to_parquet.py kept
├── *.ipynb                         # stay at root as documentation; pipeline notebooks get banner
├── tests/                          # pytest; synthetic fixtures only — never real CERT data
├── docs/                           # CLEANUP_REPORT.md (this file), PIPELINE.md (Phase 5)
├── .github/workflows/ci.yml
└── legacy/                         # gitignored; v1–v5 artifacts relocated here (Phase 6)
```

**Invariant:** the active v6 artifact directories do **not** move, so `config.py` paths
remain valid with zero changes, and the Streamlit Cloud deployment contract
(`dashboard/app.py` path, root `requirements.txt`, `runtime.txt`) is preserved throughout.

---

## 5. Phased roadmap

Each phase is an independently mergeable branch/PR, ordered low-risk → high-risk.

| Phase | Branch | Goal | Risk |
|---|---|---|---|
| 1 | `chore/cleanup-report` | This report + zero-risk git hygiene | None |
| 2 | `chore/ci-and-tests` | Ruff + pytest + GitHub Actions on the current layout | Low |
| 3 | `fix/risk-band-unification` | One canonical risk-band module [gap 1] | Med-low |
| 4 | `refactor/src-package` | `pyproject.toml` + `src/ueba/` move with back-compat shims | **Highest** |
| 5 | `feat/pipeline-cli` | `python -m ueba.pipeline <stage>` + manifest + gaps 2/3/6 | Medium |
| 6 | `chore/legacy-artifacts` | Move v1–v5 artifacts to `legacy/`; README/CLAUDE.md rewrite | Medium |
| 7 | *(deferred)* | Dashboard decomposition [gap 7] | High — own PR series |

### Phase 1 — report + hygiene *(this PR)*
- Add this report.
- `git rm --cached` the tracked `.pyc` and `dashboard/alert_state.db` (files stay on disk);
  `git rm Dashboard.html`.
- `.gitignore`: add `legacy/`, `.pytest_cache/`, `.ruff_cache/`, `dist/`, `*.egg-info/`, `.coverage`.
- CLAUDE.md: document the WebSocket localhost-only trust model; fix stale v5/CSV references.

### Phase 2 — CI + first tests (no file moves)
- `pyproject.toml` containing **only** `[tool.ruff]` + `[tool.pytest.ini_options]`
  (no `[project]` table yet → pip/Streamlit Cloud unaffected).
- `tests/conftest.py` with seeded synthetic DataFrame builders (~10 users × 30 days);
  first unit tests: `chronological_split`, `get_insiders`/`build_insider_mask`,
  `to_model_matrix`, `AlertObjectBuilder.compute_*_percentile`, `db.py` upsert
  (small refactor: `db.py` accepts an optional db path for `tmp_path` testing).
- Fix the four bare-except sites [gap 4] (behavior-preserving: log, don't crash).
- `.github/workflows/ci.yml`: ubuntu-latest, Python 3.12.
  Jobs: **lint** (`ruff check .`), **test** (`pytest -q` — pandas/numpy/pyarrow/
  scikit-learn/joblib only; **tensorflow deliberately excluded**, tests use stub models),
  **notebook guard** (nbformat validation + >15 MB size limit; no output-stripping —
  committed outputs are the documentation).

### Phase 3 — risk-band unification [gap 1]
- New `scripts/risk_bands.py` (relocates to `src/ueba/risk.py` in Phase 4):
  `BAND_ORDER`, `BAND_COLORS`, `DEFAULT_PERCENTILE_THRESHOLDS`,
  `assign_band_from_percentile()` (canonical `>=`), `assign_band_from_score()`,
  `percentile_rank()` (searchsorted).
- `AlertObjectBuilder` delegates; `live_simulation.py` drops its `_FALLBACK_*` constants.
- **Documented behavior change:** at exact boundaries the offline path changes
  (p = 95.0: HIGH → CRITICAL; p = 90.0: MEDIUM → HIGH; p = 80.0: LOW → MEDIUM).
  On-disk alert tables are unaffected until the pipeline is rerun.
- Boundary unit tests pin the unified semantics.

### Phase 4 — installable package (mechanical move, no logic changes)
- Full `[project]` table: name `ueba`, `requires-python >= 3.12`, core deps as ranges that
  admit the Cloud pins; extras `[ml]`, `[dashboard]`, `[dev]`.
- `git mv`/split per the target layout. `Preprocessing.py` (1,964 lines) splits into
  `features/{layer_a,layer_b,splits,work_hours}.py` + `constants.py` [gap 5].
  `UEBAIsolationForest.py` splits into `models/isolation_forest.py` + `models/evaluation.py`.
- **BASE_DIR rule:** `src/ueba/config.py` must not derive paths from `__file__`'s directory;
  it uses `UEBA_BASE_DIR` env override, else walks up to the directory containing
  `pyproject.toml`/`.git`, else raises with an actionable message.
- Root shims (`config.py`, `prepare_data.py`, `live_simulation.py`, `live_replay.py`, each
  moved `scripts/*.py`): `try: from ueba.X import *` with a `src/`-on-sys.path fallback —
  the dashboard and all notebooks keep working whether or not the package is installed.
- `requirements.txt` gains `-e .` (existing pins untouched — they encode a prior
  Cloud-outage fix and stay human-controlled).
- **Merge gate:** a staging Streamlit Cloud app pointed at the branch must build and boot
  before merging. Rollback: revert merge, or drop the `-e .` line (shim fallback keeps the
  app alive).

### Phase 5 — CLI pipeline + correctness fixes [gaps 2, 3, 6]
- `python -m ueba.pipeline <stage>`: `preprocess` → `train-ae` → `train-if` →
  `explain --split {calibration,test_stream,train}` → `build-alerts --split {main,calib,test}`
  → `build-dashboard`, plus `all` and `status`. `--version N` overrides the model version
  for the invocation (env-based, mirrors the existing `paths.local.py` design).
- `pipeline/manifest.py`: `require()` fail-fasts with the producing stage's name;
  `record()` writes artifact metadata (size, hash prefix, git commit, stage, version) to
  `pipeline_manifest.json`; `status` re-validates everything. The Gap-3 silent skip becomes
  a hard error (`--allow-missing` escape hatch).
  *Motivating example found during Phase 3:* `calibration_thresholds.json` does not exist
  at `CALIBRATION_THRESHOLD_PATH` on the dev machine, so the live scorer silently runs on
  fallback percentile thresholds and a fresh Alert_Object_Builder run would fail at the
  threshold-loading cell. The manifest must surface exactly this condition.
- Evaluation metrics written as JSON beside the PNGs [gap 6].
- `apply_off_hours_flags()` + injectable `LiveScorer` dependencies [gap 2]; default flags
  keep today's output byte-identical.
- Generate the missing `alert_table_6_test.parquet` [gap 3].
- Banner cell in the 5 pipeline-track notebooks pointing to the CLI; notebooks are kept,
  not deleted. New `docs/PIPELINE.md` documents the stage contracts.

### Phase 6 — legacy artifact relocation + docs refresh
- Grep gate first: no `*.py` outside docs/notebooks may reference v1–v5 paths.
- Same-volume moves (instant renames): v1–v5 datasets/models/alert tables/reconstruction
  tables → `legacy/` (gitignored; **nothing deleted**, disk stays ~36 GB). The handful of
  *tracked* v1-era files (Loss.png, notes.txt) move via `git mv`.
- Delete superseded v5-era *code* (`scripts/build_merged_parquet.py`,
  `scripts/upload_merged_parquet.py`) — recoverable from git history.
- Rewrite README.md and CLAUDE.md for the new layout and commands.

---

## 6. Cross-cutting decisions

**Dependency strategy.** `requirements.txt` remains the hand-pinned Streamlit Cloud
lockfile — never auto-generated. `pyproject.toml` owns everything else through extras:
`pip install -e .[ml]` (training), `.[dashboard]` (local dashboard dev), `.[dev]`
(tests/lint). CI never installs tensorflow; model-dependent tests use stub objects.

**Notebooks.** They stay, as documentation and analysis surfaces. The production path
becomes the CLI. No notebook execution in CI (data + runtime cost); a lightweight
nbformat/size guard keeps them healthy.

**Versioning.** The existing `MODEL_VERSION`-keyed path derivation in `config.py` is good
design and is preserved; the pipeline CLI builds on it rather than replacing it.

## 7. Explicitly out of scope (candidate follow-ups)

- Dashboard decomposition (deferred Phase 7 — `dashboard/lib/` + `pages/`, magic numbers
  to config, shared `ueba.risk.BAND_COLORS`).
- Real streaming/SIEM ingestion; scheduled retraining; model registry.
- WebSocket or Firebase auth changes.
- Model retraining or regenerating v6 artifacts (except the missing test alert table).
- Deleting any data artifacts.
- DVC/artifact version control, mypy type-checking gates, pre-commit hooks.

## 8. End-to-end verification (after all phases)

1. Fresh venv: `pip install -e .[ml,dev]` → `pytest -q` green → `ruff check .` clean.
2. `python -m ueba.pipeline status` — all v6 artifacts validated.
3. `streamlit run dashboard/app.py` — loads; all four tabs render.
4. `python live_simulation.py --interval 0.5` — scores stream to JSONL + WebSocket;
   risk bands match a pre-cleanup capture.
5. CI green on a no-op PR; staging Streamlit Cloud build succeeded on the Phase 4 branch.
6. Each notebook's import cells execute (back-compat shims resolve).
