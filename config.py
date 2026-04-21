# Central path configuration for the UEBA project.
#
# All path constants used across notebooks, scripts, and the dashboard are
# defined here. Per-contributor overrides live in paths.local.py (gitignored).
# See paths.local.example.py for the full list of overridable variables.
#
# USAGE IN NOTEBOOKS:
#   from config import CERT_PATH, UEBA_PATH, SAVE_ENCODER_PATH  # etc.
#
# USAGE IN SCRIPTS:
#   import config
#   scaler = joblib.load(config.SCALER_PATH)
#
# OVERRIDING A PATH:
#   Edit paths.local.py (copy from paths.local.example.py) and set any of:
#     CERT_PATH, MODEL_VERSION, LIVE_MODEL_VERSION,
#     ANALYST_TABLE, UEBA_DATASET

import importlib.util as _ilu
import os

# Project Root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load "paths.local.py"
_local = None
_local_path = os.path.join(BASE_DIR, "paths.local.py")
if os.path.exists(_local_path):
    _spec = _ilu.spec_from_file_location("paths_local", _local_path)
    _local = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_local)


def _local_or(attr: str, default):
    """Return the override from paths.local.py if set, otherwise *default*."""
    val = getattr(_local, attr, None) if _local is not None else None
    return val if val is not None else default


# External Dataset Root (must be set in paths.local.py)
CERT_PATH = _local_or("CERT_PATH", None)


# Insiders path derived from CERT_PATH (also overridable independently)
_insiders_default = os.path.join(CERT_PATH, "answers", "insiders.csv") if CERT_PATH else None
INSIDERS_PATH = _local_or("INSIDERS_PATH", _insiders_default)


# Active Model Version
MODEL_VERSION = _local_or("MODEL_VERSION", "5")
LIVE_MODEL_VERSION = _local_or("LIVE_MODEL_VERSION", _local_or("MODEL_VERSION", "5"))


# Version-Scoped Internal Paths
# All paths below are relative to BASE_DIR and derived from MODEL_VERSION
V = MODEL_VERSION
LV = LIVE_MODEL_VERSION


# Training Dataset
UEBA_PATH = _local_or(
    "UEBA_PATH",
    os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}_train.csv"),
)


# Preprocessing Output Directory (notebook: CERT_Preprocessing)
DEFAULT_OUTPUT_DIR = _local_or(
    "DEFAULT_OUTPUT_DIR",
    f"ueba_dataset_{V}",
)


# Encoder Artifacts
SAVE_ENCODER_PATH = _local_or(
    "SAVE_ENCODER_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{V}"),
)
ENCODER_PATH = _local_or(
    "ENCODER_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{V}", "encoder_model.keras"),
)
SCALER_PATH = _local_or(
    "SCALER_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{V}", "feature_scaler.pkl"),
)
AE_PATH = _local_or(
    "AE_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{V}", "autoencoder_model.keras"),
)


# Isolation Forest Artifacts
SAVE_IFOREST_PATH = _local_or(
    "SAVE_IFOREST_PATH",
    os.path.join(BASE_DIR, "isolation_forests", f"iforest_model_{V}"),
)
IF_PATH = _local_or(
    "IF_PATH",
    os.path.join(BASE_DIR, "isolation_forests", f"iforest_model_{V}", "iforest_model.pkl"),
)
IF_SCORES_PATH = _local_or(
    "IF_SCORES_PATH",
    os.path.join(BASE_DIR, "isolation_forests", f"iforest_model_{V}", "anomaly_scores.npy"),
)
SCORES_PATH = IF_SCORES_PATH  # alias used by Alert_Object_Builder


# Explainability Outputs
RECON_TABLE_PATH = _local_or(
    "RECON_TABLE_PATH",
    os.path.join(
        BASE_DIR, "explainability", "reconstruction_error",
        f"reconstruction_error_table_{V}.parquet",
    ),
)

# Live-Simulation Paths (uses LIVE_MODEL_VERSION)
LIVE_ENCODER_PATH = _local_or(
    "LIVE_ENCODER_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{LV}", "encoder_model.keras"),
)
LIVE_SCALER_PATH = _local_or(
    "LIVE_SCALER_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{LV}", "feature_scaler.pkl"),
)
LIVE_IF_PATH = _local_or(
    "LIVE_IF_PATH",
    os.path.join(BASE_DIR, "isolation_forests", f"iforest_model_{LV}", "iforest_model.pkl"),
)
LIVE_IF_SCORES_PATH = _local_or(
    "LIVE_IF_SCORES_PATH",
    os.path.join(BASE_DIR, "isolation_forests", f"iforest_model_{LV}", "anomaly_scores.npy"),
)
LIVE_AE_PATH = _local_or(
    "LIVE_AE_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{LV}", "autoencoder_model.keras"),
)
_live_input_parquet = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{LV}", f"ueba_dataset_{LV}_test_stream.parquet")
_live_input_csv     = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{LV}", f"ueba_dataset_{LV}_test_stream.csv")
LIVE_DEFAULT_INPUT = _local_or(
    "LIVE_DEFAULT_INPUT",
    _live_input_parquet if os.path.exists(_live_input_parquet) else _live_input_csv,
)


# Runtime Outputs (not version-scoped)
LIVE_OUTPUT = os.path.join(BASE_DIR, "processed_datasets", "live_results.jsonl")
LIVE_PAUSE_FLAG = os.path.join(BASE_DIR, "processed_datasets", "live_pause.flag")
LIVE_SIM_SCRIPT = os.path.join(BASE_DIR, "live_simulation.py")


# Dashboard Paths (parquet preferred, CSV fallback)
_analyst_override = _local_or("ANALYST_TABLE", "")
if _analyst_override:
    ANALYST_TABLE_PARQUET = _analyst_override if _analyst_override.endswith(".parquet") else ""
    ANALYST_TABLE_CSV = _analyst_override if _analyst_override.endswith(".csv") else _analyst_override
else:
    ANALYST_TABLE_PARQUET = os.path.join(BASE_DIR, "explainability", "alert_table", f"alert_table_{V}.parquet")
    ANALYST_TABLE_CSV = os.path.join(BASE_DIR, "explainability", "alert_table", f"alert_table_{V}.csv")

_ueba_override = _local_or("UEBA_DATASET", "")
if _ueba_override:
    UEBA_PARQUET = _ueba_override if _ueba_override.endswith(".parquet") else ""
    UEBA_CSV = _ueba_override if _ueba_override.endswith(".csv")     else _ueba_override
else:
    UEBA_PARQUET = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}_train.parquet")
    UEBA_CSV = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}_train.csv")

# UEBA Table A (PC-level drill-down, user/pc/day granularity)
UEBA_A_PARQUET = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}a.parquet")
UEBA_A_CSV = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}a.csv")
