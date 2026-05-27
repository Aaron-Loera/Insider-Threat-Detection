# UEBA Dataset v6 — Change Report

A detailed audit of the modifications made between `ueba_dataset_5` and
`ueba_dataset_6`. This report documents how existing features were improved,
how new computed features were added, and the motivation behind each change in
terms of downstream model behavior (Autoencoder + Isolation Forest).

---

## 0. Headline Numbers

| Metric                          | v5         | v6           |
|---------------------------------|------------|--------------|
| Layer A features                | 37         | 54           |
| Layer B features                | 110        | 414          |
| Off-hours definition            | Static (9, 17) for all users | Per-user envelope from history, with (9, 17) fallback |
| Primary z-score window          | Unbounded expanding | 30-day rolling (min 14 days history) |
| Long-horizon z-score            | —          | 90-day rolling (min 30 days history) |
| Peer-group baseline             | —          | Leave-one-out z-scores against LDAP role |
| Calibration slice               | —          | Insider-free held-out 10% slice |
| File-channel ingestion          | In-memory  | Chunked (memory-safe) |
| Working-hour fallback           | (9, 17)    | (9, 17) |
| Rolling window (legacy delta)   | 5 days     | 5 days |

---

## 1. Improvements to Existing Features

### 1.1 Per-user working-hour envelopes (replaces static (9, 17) off-hours flags)

**What changed.** A new helper `compute_user_work_hours()` derives a personal
`(start_hour, end_hour)` for each user from the 10th and 90th percentile of
that user's historical Logon-event hours. Users with fewer than 30 prior
logon-days fall back to the population default `(9, 17)` and are tagged
`schedule_complete=False`. Every channel extractor (`extract_logon_features`,
`extract_file_features[_chunked]`, `extract_device_features`,
`extract_email_features_chunked`, `extract_http_features_chunked`) now accepts
a `user_work_hours` argument and uses `_compute_off_hours()` to derive the
`off_hours` flag per row from that personalized envelope.

The derived schedule is persisted to
`processed_datasets/ueba_dataset_6/user_work_hours.parquet` so
`live_simulation.py` and any future retraining can re-apply the same envelope.

**Why this matters for modeling.** v5 used a single `(9, 17)` window for the
entire population. A 7am-start engineer and a 10am-start analyst produced
wildly different baseline off-hour counts that had nothing to do with
threat-relevant behavior — the model had to learn to ignore predictable
shift differences before it could learn anything else. Personalizing the
envelope means an off-hour flag now indicates *deviation from this user's
own schedule*, which is what the AE actually needs to reconstruct accurately.
In the v6 preprocessing run, 2,474 of 4,000 users received a personal
schedule; the remaining low-history users use the fallback and are gated by
`schedule_complete`.

### 1.2 Bounded trailing z-scores (replaces unbounded expanding window)

**What changed.** `apply_ueba_enhancements()` now computes per-user z-scores
over a 30-day rolling window (`zscore_window=30`,
`zscore_min_history=14`) instead of the v5 expanding window. The old
expanding mean/std grew without bound as a user accumulated history.

**Why this matters for modeling.** Under the v5 expanding baseline, an
insider with sustained escalation (the canonical example in the project
being `CDE1846`) was absorbed into their own baseline within ~10 days: every
new anomalous day raised the mean and inflated the std, shrinking the
z-score back toward zero. A bounded 30-day window keeps the baseline
stationary enough that sustained behavioral shifts remain visible to the
autoencoder for the full duration of the drift, not just the first week of it.

### 1.3 `baseline_complete` gate (new column, safeguards every derived signal)

**What changed.** A boolean column `baseline_complete` is appended in
`apply_ueba_enhancements()`, set `True` once a user has at least
`zscore_min_history` prior observations. Downstream risk banding is expected
to refuse promotion to CRITICAL while `baseline_complete=False`.

**Why this matters for modeling.** New users with thin history previously
produced artificially extreme z-scores on their first active days (small
denominators → large quotients). The gate prevents this "cold-start spike"
failure mode from generating false-positive alerts.

### 1.4 Calibration split (replaces single 90/10 split)

**What changed.** Layer B is now split chronologically into three slices
instead of two, producing four downstream parquet artifacts:

| Artifact                                       | Share | Insiders | Purpose                                    |
|------------------------------------------------|-------|----------|--------------------------------------------|
| `ueba_dataset_6_train.parquet`                 | ~80%  | retained (used to fit AE on normal-only via mask) | Model training |
| `ueba_dataset_6_calibration.parquet`           | ~10%  | removed  | Fit AE/IF baselines, calibrate thresholds  |
| `ueba_dataset_6_calibration_eval.parquet`      | ~10%  | retained | Held-out AE/IF evaluation only             |
| `ueba_dataset_6_test_stream.parquet`           | ~10%  | retained | Live simulation input                      |

**Why this matters for modeling.** v5 banded risk against the *training*
score distribution, which the AE had already minimized — that produced
optimistic percentile thresholds. v6 calibrates against a held-out,
insider-free slice the model never trained on, so the percentile bands
(`CRITICAL ≥ 95th`, etc.) actually mean "≥ 95th percentile of unseen normal
behavior," and the eval slice with insiders retained gives an honest
out-of-sample AUROC.

### 1.5 Memory-safe file-channel ingestion

**What changed.** `file` is now included in `LARGE_FILE_SOURCES`, and a new
`extract_file_features_chunked()` mirrors the chunked aggregation pattern
already used for email and http. The chunked variant accumulates partial
aggregations, identity frames, hourly counts, and a minimal
`(user, pc, day, timestamp)` frame for the longest-run computation, then
combines them via `combine_partial_aggregations` and `build_unique_count`.

In parallel, `normalize_shared_columns` was rewritten to apply string
normalization (`.str.lower().str.strip()`) only to the small set of unique
user/pc category codes (~4,000), not to every row. This eliminates the
large intermediate object arrays that caused `MemoryError` on multi-million-
row logs.

**Why this matters for modeling.** Enables the v6 sub-day intensity
features (Section 2.1) on the file channel without running out of memory.
This is a precondition for everything in Section 2 — without chunking, the
new entropy / peak-hour / longest-run columns could not be computed on
`file.csv`.

---

## 2. New Computed Features

### 2.1 Sub-day intensity features (per channel)

**What was added.** Three new columns per channel
(`logon`, `file`, `device`, `email`, `http`):

- `<channel>_hourly_entropy` — Shannon entropy of the user-day's hour-of-day
  distribution. Low entropy → activity concentrated in a few hours; high
  entropy → activity spread across the day.
- `<channel>_peak_hour_count` — maximum events seen in any single hour that
  day. **Skipped for the `device` channel** per the audit recommendation
  (low event volume per day makes the peak unstable).
- `<channel>_longest_active_run_minutes` — length of the longest contiguous
  session, where a session breaks at any gap > 30 minutes between events.
  Computed vectorized by `_compute_longest_run()`.

At Layer B, these are **max-aggregated across PCs** so a user-day inherits
the worst-case burst signal across all machines they touched.

**Why this matters for modeling.** v5 only exposed daily totals. A user who
normally spreads 200 file opens across 8 hours and one day compresses them
into a 45-minute burst produces the *exact same* `file_open_count` but a
very different `file_hourly_entropy` and `file_longest_active_run_minutes`.
Bulk exfiltration, automated scripts, and late-night dwell are all sub-day
phenomena that daily counts cannot see — entropy and run-length expose them
directly, giving the AE three new orthogonal axes per channel.

### 2.2 Late-night activity counters (per channel)

**What was added.** `<channel>_late_night_count` for each channel — events
occurring **22:00–04:59**, computed *independently* of the user's personal
working-hour envelope.

**Why this matters for modeling.** Per-user off-hour flags (Section 1.1)
correctly absorb shift workers and early-riser engineers — that is what we
want for *personal* drift detection. But true 2am activity is anomalous for
almost every user in this corpus regardless of their personal schedule.
Late-night counters preserve that absolute signal in parallel with the
relative off-hours signal, so the model can use both "unusual for this
user" and "unusual for any user."

### 2.3 Multi-horizon rolling features (applied to every base feature)

**What was added.** `_add_multihorizon_features()` generates three columns
for every Layer B base feature:

- `<feature>_7d_sum` — prior-7-day rolling sum (shifted by 1 day; no leakage).
- `<feature>_30d_sum` — prior-30-day rolling sum (shifted by 1 day).
- `<feature>_1d_over_30d_ratio` — `today / (30d_daily_avg + 0.5)`, clipped
  to `[0, 50]`. A ratio of 10 means the user did 10× their monthly average
  on that single day. The `+ 0.5` epsilon suppresses noise on near-zero
  baselines; the `clip` bounds AE input magnitude.

**Why this matters for modeling.** Gives the AE explicit access to "how
unusual is today relative to this month?" without forcing it to derive that
ratio from raw counts. The 7d/30d windows also let the model see
*accumulating* anomalies — e.g., a slow data-hoarding pattern over two
weeks that no single-day z-score will flag because each individual day
looks unremarkable.

### 2.4 Long-horizon (90-day) z-scores

**What was added.** `<feature>_zscore_90d` — per-user z-score against a
90-day trailing window (`longhorizon_window=90`,
`longhorizon_min_history=30`), computed in parallel with the primary
30-day z-score from Section 1.2.

**Why this matters for modeling.** Catches slow drift that outruns the
30-day baseline. The two horizons together let the model distinguish a
one-day spike (high 30d-z, moderate 90d-z) from a sustained shift (both
elevated) — two patterns that are operationally very different and that
the v5 single-horizon z-score collapsed into one signal.

### 2.5 Peer-group z-scores (LDAP-driven)

**What was added.** A new ingestion step and enhancement pass:

- `load_ldap()` reads CERT's monthly LDAP snapshot files
  (`LDAP/YYYY-MM.csv`) and **retains all snapshot rows** without deduplication.
  This design preserves the full snapshot history so that `build_user_profiles()`
  can compute the `is_active` flag from snapshot presence. The function
  returns `[user, employee_name, role, department, team, functional_unit,
  supervisor, _snapshot]`, where `_snapshot` is a `pd.Timestamp` parsed from
  the filename (`YYYY-MM`).
- `apply_peer_group_enhancements()` computes a leave-one-out z-score for
  each `(peer_group, day)` cohort: every base feature gets a
  `<feature>_peer_zscore` column standardized against the user's peers,
  clipped to `[-10, 10]`. Before mapping peer groups, it collapses `ldap_df`
  to one row per user (latest snapshot wins) internally. Peer grouping is
  configurable via `PEER_GROUP_KEY` in `config.py` (default `role`,
  swappable to `department` or `team`). Leave-one-out variance is computed
  via the variance-decomposition trick to avoid an expensive per-group apply.

**Why this matters for modeling.** The per-user z-score answers "is this
unusual for *this* user?" — but a user who has been compromised for weeks
will have a corrupted baseline, and more of the same malicious behavior
will register as a z-score near zero. The peer-group z-score answers "is
this unusual for someone in this role?" — which a single user's history
cannot mask. Combining the two is what unlocks detection of insiders whose
entire baseline is the anomaly, not just users whose recent activity
deviates from their own past.

### 2.6 User profile enrichment (LDAP identity columns + role sensitivity)

**What was added.** After peer-group z-scores are applied, `build_layer_b()`
calls two additional LDAP helpers:

- `build_user_profiles(ldap_raw)` collapses the multi-snapshot LDAP table to
  one row per user (latest snapshot wins via `groupby + last()`), then appends
  a boolean `is_active` flag — `True` iff the user appears in the most recent
  LDAP monthly file. This produces a user-level lookup with columns
  `[user, employee_name, role, department, functional_unit, supervisor, is_active]`.

- `compute_role_sensitivity(role, department)` maps each user's role and
  department to a `float32` sensitivity weight in `[0.0, 1.0]`.
  Executives, IT-admin, and finance roles receive `0.80–1.00`; standard
  individual contributors receive `0.30–0.50`. Finance departments
  (Accounting, Payroll, FinancialPlanning, Pricing) receive a floor of `0.70`
  regardless of role. Users with an unrecognised role default to `0.50`.

These two outputs are left-joined onto the final Layer B matrix, adding the
following columns to every `(user, day)` row:

| Column            | Type      | Source                        |
|-------------------|-----------|-------------------------------|
| `employee_name`   | str       | LDAP (latest snapshot)        |
| `role`            | str       | LDAP (latest snapshot)        |
| `department`      | str       | LDAP (latest snapshot)        |
| `functional_unit` | str       | LDAP (latest snapshot)        |
| `supervisor`      | str/None  | LDAP (latest snapshot)        |
| `role_sensitivity`| float32   | `compute_role_sensitivity()`  |
| `is_active`       | bool      | Snapshot presence flag        |

Missing users (not in LDAP) receive `employee_name = user`, `Unknown` for
categorical fields, `role_sensitivity = 0.5`, and `is_active = False`.

**Why this matters for modeling and triage.** `role_sensitivity` gives the
downstream risk-banding logic a principled way to weight anomaly scores
by how damaging a breach would be for that role — an IT-admin or CFO
anomaly is not equivalent to a stockroom clerk anomaly at the same raw
percentile. `is_active` enables the alert pipeline to suppress or
down-weight alerts for former employees who appear in test data due to
lag in log retention. `employee_name`, `department`, and `supervisor` are
surfaced in the dashboard's Investigation tab for analyst triage without
requiring a separate LDAP lookup at query time.

---

## 3. Pipeline Artifacts (v6)

The v6 preprocessing pipeline produces the following files under
`processed_datasets/ueba_dataset_6/`:

- `ueba_dataset_6a.csv` / `ueba_dataset_6a.parquet` — Layer A,
  `(user, pc, day)` granularity, for drill-down analysis (54 features).
- `ueba_dataset_6b.csv` / `ueba_dataset_6b.parquet` — Layer B,
  `(user, day)` model-ready matrix (407 features), including LDAP-derived
  identity columns (`employee_name`, `role`, `department`, `functional_unit`,
  `supervisor`, `role_sensitivity`, `is_active`) joined from `build_user_profiles()`.
- `ueba_dataset_6_train.parquet` — first ~80% chronologically (model training).
- `ueba_dataset_6_calibration.parquet` — middle ~10%, insiders removed,
  used to fit AE/IF baselines and threshold calibration.
- `ueba_dataset_6_calibration_eval.parquet` — middle ~10%, insiders
  retained, used only for held-out AE/IF evaluation.
- `ueba_dataset_6_test_stream.parquet` — last ~10%, fed to
  `live_simulation.py`.
- `user_work_hours.parquet` — per-user `(start_hour, end_hour,
  schedule_complete)` table, must be reapplied at inference time.

---

## 4. Expected Downstream Impact

| Capability                                                | Mechanism                                                          |
|-----------------------------------------------------------|---------------------------------------------------------------------|
| Detecting sustained insider drift (CDE1846-style)         | Bounded 30d z-score + 90d z-score + peer-group z-score              |
| Detecting bulk exfiltration bursts                        | `*_hourly_entropy`, `*_peak_hour_count`, `*_longest_active_run_minutes`, `*_1d_over_30d_ratio` |
| Detecting slow accumulation (data hoarding)               | `*_7d_sum`, `*_30d_sum`, 90d z-score                                |
| Detecting after-hours dwell that respects shift-workers   | Per-user off-hours + parallel `*_late_night_count`                  |
| Detecting users whose entire baseline is anomalous        | Peer-group z-scores against LDAP role                               |
| Reducing cold-start false positives                       | `baseline_complete` gate                                            |
| Honest threshold calibration                              | Insider-free calibration slice                                      |
| Role-weighted risk prioritization                         | `role_sensitivity` (0.3–1.0) from `compute_role_sensitivity()`      |
| Suppressing stale-employee false positives                | `is_active` flag from LDAP snapshot presence                        |
| Analyst triage without secondary LDAP lookup              | `employee_name`, `department`, `supervisor` pre-joined to Layer B   |
