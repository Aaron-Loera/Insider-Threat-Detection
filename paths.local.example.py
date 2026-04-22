# ─────────────────────────────────────────────────────────────────────────────
# paths.local.example.py  —  LOCAL PATH OVERRIDES (template)
#
# HOW TO USE:
#   1. Copy this file and rename it to:  paths.local.py
#   2. Fill in the absolute paths to wherever your data files live.
#   3. paths.local.py is gitignored — it will never be committed.
#      Every contributor maintains their own copy.
#
# WHICH PATHS TO SET:
#   Only set a variable if your file lives somewhere OTHER than the default
#   location under the project root.  You can leave any line commented out
#   to keep the default relative path.
#
# DEFAULT LOCATIONS (relative to project root):
#   explainability/alert_table/alert_table_{V}.parquet   (or .csv)
#   processed_datasets/ueba_dataset_{V}/ueba_dataset_{V}_train.parquet   (or .csv)
#   where {V} = MODEL_VERSION (default "5")
# ─────────────────────────────────────────────────────────────────────────────

# ── Raw CERT dataset root (REQUIRED for preprocessing notebooks) ──────────────
# Set this to the folder containing the CERT r6.2 CSVs on your machine.
# There is no project-relative default — every contributor must set this.
# CERT_PATH = r"C:\Users\yourname\Documents\Datasets\CERT_r6.2"

# ── Active model version ──────────────────────────────────────────────────────
# Controls which numbered model artifacts (encoder, isolation forest, alert table,
# ueba dataset) are used by default across all notebooks and scripts.
# MODEL_VERSION = "5"

# ── Live simulation model version ─────────────────────────────────────────────
# Defaults to MODEL_VERSION if not set.  Set this only if your deployed/live
# scoring model differs from the version used in the offline training notebooks.
# LIVE_MODEL_VERSION = "4"

# ── Dashboard paths ───────────────────────────────────────────────────────────
# Absolute path to the alert table produced by Alert_Object_Builder.ipynb.
# Parquet is preferred; the dashboard falls back to CSV if only that exists.
# ANALYST_TABLE = r"C:\Users\yourname\data\alert_table_5.parquet"

# Absolute path to the UEBA training dataset produced by CERT_Preprocessing.ipynb.
# UEBA_DATASET = r"C:\Users\yourname\data\ueba_dataset_5_train.parquet"
