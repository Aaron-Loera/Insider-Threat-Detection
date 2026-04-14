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
#   explainability/alert_table/alert_table_4.parquet   (or .csv)
#   processed_datasets/ueba_dataset_4/ueba_dataset_4_train.parquet   (or .csv)
# ─────────────────────────────────────────────────────────────────────────────

# Absolute path to the alert table produced by Alert_Object_Builder.ipynb.
# Parquet is preferred; the app falls back to CSV if only that exists.
# ANALYST_TABLE = r"C:\Users\yourname\data\alert_table_4.parquet"

# Absolute path to the UEBA training dataset produced by CERT_Preprocessing.ipynb.
# UEBA_DATASET = r"C:\Users\yourname\data\ueba_dataset_4_train.parquet"
