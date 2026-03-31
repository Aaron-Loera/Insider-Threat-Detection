---
name: analyze-data-integrity
description: >
  Audits Python scripts and Jupyter Notebooks for ML data integrity violations, data leakage, and
  preprocessing errors. Use this skill whenever the user asks to "check", "audit", "review",
  "analyze", or "validate" their data pipeline, preprocessing code, or ML workflow — especially
  for Autoencoder, Isolation Forest, or any anomaly detection system. Also trigger when the user
  asks about "data leakage", "train/test contamination", "normalization", "scaling", "feature
  engineering", or says something like "is my preprocessing correct?", "could my model be leaking?",
  or "is my dataset clean?". Trigger even for casual phrasing like "can you look at my pipeline" or
  "does anything look off in my data prep". Always use this skill over a generic code review when
  data quality or ML correctness is the concern.
---
 
# Skill: analyze_data_integrity
 
A thorough data integrity and leakage auditor for Python ML pipelines, with domain-specific
awareness for **Autoencoder + Isolation Forest anomaly detection** systems (e.g., the CERT
insider threat detection pipeline using TensorFlow/Keras + scikit-learn).
 
---
 
## Audit Workflow
 
### Step 1 — Read the Code
 
Use the `view` tool (or read from context if already provided) to read the full script or notebook.
Look for all stages of the ML pipeline:
 
1. **Data loading** — how raw data enters the pipeline
2. **Train/test splitting** — when and how data is partitioned
3. **Preprocessing / normalization / scaling** — `StandardScaler`, `MinMaxScaler`, `fit()`, `fit_transform()`, `transform()`
4. **Feature engineering** — derived columns, aggregations, encodings
5. **Model training** — what data is passed to `.fit()` for Autoencoder and Isolation Forest
6. **Threshold / score calibration** — how anomaly thresholds are chosen
7. **Inference pipeline** — how saved scalers/models are applied to new data
 
### Step 2 — Identify Issues
 
Check for all issues in the **Checklist** section below. Classify each finding as HIGH, MEDIUM, or LOW.
 
### Step 3 — Report in Structured Format
 
Always output the report in this exact structure:
 
```
## 🔴 HIGH RISK — [N found]
Issues that directly compromise model validity and must be fixed.
For each issue:
- **[Issue Title]** — Line X (or "throughout")
  - Problem: [In-depth explanation of what is wrong and why it matters]
  - Fix: [Concrete, working code correction or refactored snippet]
 
## 🟡 MEDIUM RISK — [N found]
Issues that may degrade model performance or reliability.
For each issue:
- **[Issue Title]** — Line X
  - Problem: [Clear explanation of the concern]
  - *(Request a fix to get a working solution)*
 
## 🟢 LOW RISK — [N found]
Best-practice gaps and minor quality concerns.
For each issue:
- **[Issue Title]** — Line X
  - Problem: [Brief note on the concern]
  - *(Request a fix to get a working solution)*
```
 
**Output rules by risk level:**
 
| Level | Description | Solution |
|-------|-------------|----------|
| 🔴 HIGH | In-depth explanation of the root cause and impact | ✅ Always provided automatically |
| 🟡 MEDIUM | Clear description of the problem | ❌ Only provide if user asks |
| 🟢 LOW | Brief note on the concern | ❌ Only provide if user asks |
 
---
 
## Issue Checklist
 
### 🔴 HIGH RISK Issues
 
#### H1 — Scaler/Encoder Fit on Full Dataset Before Split
**What to look for:** `scaler.fit(df)` or `scaler.fit_transform(df)` called on the entire
dataset *before* `train_test_split()` or any manual split.
 
**Why it matters:** Statistics computed from the full dataset (mean, std, min, max) incorporate
information from future/test samples. The model implicitly "sees" test data distribution during
training, inflating performance metrics and producing an unreliable model.
 
**Fix pattern:**
```python
# WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fitted on all data
X_train, X_test = train_test_split(X_scaled, ...)
 
# CORRECT
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit ONLY on train
X_test_scaled = scaler.transform(X_test)         # transform only
```
 
---
 
#### H2 — Autoencoder Trained on Anomalous Samples
**What to look for:** Autoencoder `.fit()` receives a dataset that includes labeled anomalous
records, or the training set is not filtered to "normal only" behavior before Autoencoder training.
 
**Why it matters:** Autoencoders for anomaly detection rely on *reconstruction error* to flag
anomalies. If anomalous behavior is present during training, the model learns to reconstruct it
too — destroying the anomaly signal. The Autoencoder should be trained **exclusively on normal
behavior** so that anomalous inputs produce elevated reconstruction error.
 
**Fix pattern:**
```python
# WRONG — trains on mixed data
autoencoder.fit(X_train_scaled, X_train_scaled, ...)
 
# CORRECT — filter to normal class only (label == 0 in CERT dataset)
X_train_normal = X_train_scaled[y_train == 0]
autoencoder.fit(X_train_normal, X_train_normal, ...)
```
 
---
 
#### H3 — Test Set Used to Tune Anomaly Threshold
**What to look for:** Reconstruction error threshold or Isolation Forest `contamination` is
calibrated using `X_test`, `y_test`, or metrics derived from the test set.
 
**Why it matters:** Selecting a threshold that maximizes F1/accuracy on the test set is a form of
data leakage — the threshold is overfit to that specific test split. In production the threshold
will underperform.
 
**Fix pattern:**
```python
# WRONG — threshold decided by peeking at test performance
errors = reconstruction_error(X_test)
threshold = np.percentile(errors, 95)
 
# CORRECT — derive threshold from validation set or training errors only
val_errors = reconstruction_error(X_val)   # or from train normals
threshold = np.percentile(val_errors, 95)
```
 
---
 
#### H4 — Isolation Forest Fitted on Test Embeddings
**What to look for:** `IsolationForest.fit()` is called on latent embeddings that include
test-set records (i.e., `iso_forest.fit(all_embeddings)` instead of train embeddings only).
 
**Why it matters:** The Isolation Forest learns partitioning boundaries from *all available data*,
including test records. This leaks test-set distribution into the model and invalidates generalization
testing entirely.
 
**Fix pattern:**
```python
# WRONG
all_embeddings = encoder.predict(X_all_scaled)
iso_forest.fit(all_embeddings)
 
# CORRECT
train_embeddings = encoder.predict(X_train_scaled)
iso_forest.fit(train_embeddings)
test_embeddings = encoder.predict(X_test_scaled)
scores = iso_forest.decision_function(test_embeddings)
```
 
---
 
#### H5 — Label / Target Column Included in Feature Matrix
**What to look for:** `insider` flag, threat label, or any derived column directly encoding
ground truth is present in `X` or the scaled feature array passed to the model.
 
**Why it matters:** Including the target in the feature set trivially leaks the answer to the
model. Even if dropped "at the last moment", intermediate transformations (e.g., correlation-based
feature selection) may have already used the label.
 
**Fix pattern:**
```python
# WRONG
X = df.drop(columns=[])  # forgot to drop label column 'insider'
 
# CORRECT
LABEL_COL = 'insider'
X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]
```
 
---
 
#### H6 — Preprocessing Pipeline Not Persisted for Inference
**What to look for:** Scaler/encoder is `fit_transform`'d at training time but not saved (e.g.,
no `joblib.dump(scaler, ...)` or `pickle`). At inference, a *new* scaler is fit on incoming
streaming data.
 
**Why it matters:** Re-fitting on streaming data produces a different feature space than what the
Autoencoder/Isolation Forest was trained on. The model will receive out-of-distribution inputs at
inference time — leading to silent failures or inflated anomaly scores.
 
**Fix pattern:**
```python
import joblib
 
# At training time — save the fitted scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'models/scaler.pkl')
 
# At inference time — load and reuse
scaler = joblib.load('models/scaler.pkl')
X_new_scaled = scaler.transform(X_new)   # DO NOT re-fit
```
 
---
 
### 🟡 MEDIUM RISK Issues
 
#### M1 — Imputation / Missing Value Fill Applied Before Split
**What to look for:** `df.fillna(df.mean())`, `SimpleImputer.fit_transform(df)`, or `df.interpolate()`
applied to the full dataframe before train/test splitting.
 
**Why it matters:** Mean/median imputation statistics derived from the full dataset include test
set values, introducing a mild but measurable leakage of distribution information.
 
---
 
#### M2 — Feature Selection / Correlation Filter Uses Full Dataset
**What to look for:** `df.corr()`, `SelectKBest`, `VarianceThreshold`, or manual column
drops based on correlation computed on all rows before splitting.
 
**Why it matters:** Features selected using all data may retain test-set signal, creating selection
bias that overstates model performance.
 
---
 
#### M3 — No Stratification on Imbalanced Classes
**What to look for:** `train_test_split(X, y)` without `stratify=y` on a dataset where
anomalous records are rare (as in CERT, where insider events are <5% of records).
 
**Why it matters:** Without stratification, the test set may contain disproportionately few (or
many) anomalous samples, making evaluation metrics unreliable and unstable across random seeds.
 
---
 
#### M4 — Isolation Forest `contamination` Set to Default Without Justification
**What to look for:** `IsolationForest()` with no `contamination` argument, or
`contamination='auto'` with no comment on expected anomaly rate.
 
**Why it matters:** The `contamination` parameter defines the expected proportion of anomalies
and directly influences the decision threshold. On CERT, the true anomaly rate is known — using
a data-informed value significantly improves precision.
 
---
 
#### M5 — Validation Split Absent for Autoencoder Early Stopping
**What to look for:** `autoencoder.fit(X_train, X_train, epochs=N)` with no
`validation_data` or `validation_split` argument.
 
**Why it matters:** Without a validation split, early stopping (`EarlyStopping` callback)
cannot fire — and without early stopping, the Autoencoder may overfit to normal training samples,
reducing its ability to flag genuinely anomalous inputs.
 
---
 
#### M6 — Train/Test Overlap in Time-Ordered Data
**What to look for:** Random shuffling (`shuffle=True` default in `train_test_split`) applied
to a dataset that is naturally time-ordered (logs sorted by `date`, `timestamp`, or `week`).
 
**Why it matters:** For time-series behavioral logs (CERT is ordered by date), random splitting
introduces temporal leakage — the model is trained on *future* events and tested on *past* ones,
which is unrealistic for deployment.
 
---
 
#### M7 — Autoencoder Reconstruction Threshold Applied Globally Across Users
**What to look for:** A single scalar threshold applied to reconstruction errors for all users,
with no per-user or per-role baseline.
 
**Why it matters:** Different users have different behavioral baselines. A global threshold
produces high false positive rates for power users (whose normal behavior looks anomalous
globally) and misses threats from low-activity accounts.
 
---
 
### 🟢 LOW RISK Issues
 
#### L1 — No Random Seed Set
**What to look for:** `train_test_split`, `IsolationForest`, `np.random`, or TensorFlow
model initialization without `random_state` / `tf.random.set_seed()`.
 
**Why it matters:** Non-reproducible splits and model initialization make debugging, comparison,
and academic evaluation unreliable.
 
---
 
#### L2 — Scaling Strategy Not Justified
**What to look for:** `StandardScaler` or `MinMaxScaler` used without a comment or
documented rationale for the choice.
 
**Why it matters:** Autoencoders are sensitive to input range. `MinMaxScaler` is typically
preferred for bounded inputs (0–1) fed into sigmoid-output Autoencoders. `StandardScaler` suits
Gaussian-distributed features. A mismatch can slow convergence.
 
---
 
#### L3 — No Duplicate Record Check Between Train and Test
**What to look for:** No deduplication step (`df.drop_duplicates()`) and no check that
`X_train` and `X_test` share no identical rows.
 
**Why it matters:** Duplicate records that appear in both train and test sets cause optimistic
evaluation — the model is literally tested on data it has seen.
 
---
 
#### L4 — Preprocessing Steps Not Documented in Code
**What to look for:** Scaling, encoding, and feature selection steps applied silently with
no comments explaining *why* each step is taken.
 
**Why it matters:** Undocumented pipelines are difficult to audit, replicate, and hand off.
For a capstone/SOC system, explainability of the preprocessing chain matters as much as the
model itself.
 
---
 
#### L5 — No Assertion on Feature Dimensionality Consistency
**What to look for:** No `assert X_train.shape[1] == X_test.shape[1]` or schema validation
before model training or inference.
 
**Why it matters:** Silent shape mismatches can cause cryptic runtime errors or — worse —
silently pass when columns are reordered, producing garbage predictions.
 
---
 
#### L6 — No Check for NaN/Inf Values After Feature Engineering
**What to look for:** No `assert not df.isnull().any().any()` or `np.isfinite()` check
after all transformations, before training.
 
**Why it matters:** NaN or infinite values passed to the Autoencoder or Isolation Forest
produce silent NaN gradients (TF) or incorrect tree partitions (sklearn), corrupting the model.
 
---
 
## Quick Reference: Common Patterns by Framework
 
### TensorFlow / Keras (Autoencoder)
- Fit scaler BEFORE passing to `.fit()` — and on train-normal only
- Use `validation_data=(X_val, X_val)` with `EarlyStopping`
- Save with `model.save('autoencoder.h5')` and `joblib.dump(scaler, 'scaler.pkl')`
- At inference: `scaler.transform()` (never `fit_transform`) → `model.predict()` → compute MSE
 
### Scikit-learn (Isolation Forest)
- Pass `contamination=` informed by known anomaly rate in training data
- Fit **only** on normal train embeddings
- Use `decision_function()` for a continuous score; `predict()` for binary label
- Set `random_state=42` for reproducibility
 
### Pandas (Feature Engineering)
- Apply all `groupby`, `merge`, `fillna` operations on train set, then join to test
- Never compute aggregate statistics (mean, std, percentile) across the full df before splitting
 
---
 
## Notes on This Project
 
This skill is tuned for the **Team DSK Insider Threat Detection** system, which uses:
- **CERT synthetic dataset** — time-ordered, class-imbalanced (~<5% anomalies)
- **Autoencoder** (Keras) — should train on normal-only behavioral embeddings
- **Isolation Forest** (sklearn) — anomaly detection on latent space
- **Streamlit** — inference pipeline must use *persisted* scalers and models
 
Pay particular attention to H2 (Autoencoder trained on anomalous data) and H6 (scaler not
persisted), as these are the most commonly encountered issues in this exact architecture.