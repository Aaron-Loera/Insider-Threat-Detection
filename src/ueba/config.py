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
#     ANALYST_TABLE, UEBA_DATASET, UEBA_CALIBRATION_PATH,
#     AE_BASELINE_PATH, IF_BASELINE_PATH, CALIBRATION_THRESHOLD_PATH

import importlib.util as _ilu
import os


def _find_base_dir() -> str:
    """Resolve the repository root.

    This module lives at src/ueba/config.py, so its own directory is NOT the
    project root. Resolution order:
      1. UEBA_BASE_DIR environment variable (set this for non-editable installs
         or when running against a relocated data tree),
      2. walk upward from this file until a directory containing
         pyproject.toml or .git is found (always succeeds for editable
         installs and source checkouts).
    """
    env = os.environ.get("UEBA_BASE_DIR")
    if env:
        return os.path.abspath(env)
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(d, "pyproject.toml")) or os.path.exists(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:  # filesystem root reached
            raise RuntimeError(
                "ueba.config could not locate the project root: no pyproject.toml or .git "
                "found in any parent directory. Set the UEBA_BASE_DIR environment variable "
                "to the repository root (the directory containing processed_datasets/)."
            )
        d = parent


# Project Root
BASE_DIR = _find_base_dir()


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
MODEL_VERSION = _local_or("MODEL_VERSION", "6")
LIVE_MODEL_VERSION = _local_or("LIVE_MODEL_VERSION", "6")

# Peer-group column used by apply_peer_group_enhancements (see scripts/Preprocessing.py).
# Swappable to "department" or "team" once role granularity is evaluated.
PEER_GROUP_KEY = _local_or("PEER_GROUP_KEY", "role")


# Columns that are NOT model features: identifiers, LDAP profile/identity
# enrichment, and gating flags. Everything else in the v6 matrix is a numeric
# behavioral feature fed to the scaler + autoencoder + isolation forest.
#
# The role/department/functional_unit signal is already represented to the
# models via the *_peer_zscore features, so the raw categorical columns are
# excluded (including them would inject rare-category-as-anomaly bias). The
# identity columns (employee_name, supervisor) would leak identity, and the
# static per-user signals (is_active, role_sensitivity) belong in the alert
# layer, not the unsupervised anomaly score. "pc" is listed so the same helper
# can be reused on (user, pc, day) frames; it is filtered to existing columns.
NON_FEATURE_COLS = [
    "user", "day", "pc",                      # identifiers
    "employee_name", "supervisor",            # identity (leakage)
    "role", "department", "functional_unit",  # org context (covered by *_peer_zscore)
    "is_active",                              # employment status (alert-layer gate)
    "role_sensitivity",                       # triage weight (alert-layer)
    "baseline_complete",                      # cold-start gate
]


# Version-Scoped Internal Paths
# All paths below are relative to BASE_DIR and derived from MODEL_VERSION
V = MODEL_VERSION
LV = LIVE_MODEL_VERSION


# Training Dataset
UEBA_PATH = _local_or(
    "UEBA_PATH",
    os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}_train.parquet"),
)

# Calibration slice (held-out 10%, insider-free — used for baseline fitting and threshold calibration)
UEBA_CALIBRATION_PATH = _local_or(
    "UEBA_CALIBRATION_PATH",
    os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}_calibration.parquet"),
)

# Calibration eval slice (same time window as UEBA_CALIBRATION_PATH but retains insider users;
# used only for held-out AE/IF evaluation — never for baseline fitting)
UEBA_CALIB_EVAL_PATH = _local_or(
    "UEBA_CALIB_EVAL_PATH",
    os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}_calibration_eval.parquet"),
)


# Preprocessing Output Directory (notebook: CERT_Preprocessing)
DEFAULT_OUTPUT_DIR = _local_or(
    "DEFAULT_OUTPUT_DIR",
    os.path.join("processed_datasets", f"ueba_dataset_{V}")
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

# Clean calibration baseline — AE reconstruction errors from the insider-free calibration slice
AE_BASELINE_PATH = _local_or(
    "AE_BASELINE_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{V}", "ae_baseline_clean.npy"),
)

# Absolute risk-band thresholds calibrated against the clean calibration baseline
CALIBRATION_THRESHOLD_PATH = _local_or(
    "CALIBRATION_THRESHOLD_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{V}", "calibration_thresholds.json"),
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

# Clean calibration baseline — IF anomaly scores from the insider-free calibration slice
IF_BASELINE_PATH = _local_or(
    "IF_BASELINE_PATH",
    os.path.join(BASE_DIR, "isolation_forests", f"iforest_model_{V}", "if_baseline_clean.npy"),
)


# Explainability Outputs
RECON_TABLE_PATH = _local_or(
    "RECON_TABLE_PATH",
    os.path.join(
        BASE_DIR, "explainability", "reconstruction_error",
        f"reconstruction_error_table_{V}.parquet",
    ),
)

# Peer-group baselines — (department, day, <behavioral features>) parquet used by
# the Investigation tab to render peer-comparison charts.  Generated from the
# training dataset by grouping on (department, day) and computing column means.
# The dashboard degrades gracefully (peer comparison disabled) when this file
# does not yet exist.
PEER_BASELINES_PATH = _local_or(
    "PEER_BASELINES_PATH",
    os.path.join(
        BASE_DIR, "processed_datasets", f"ueba_dataset_{V}",
        f"peer_baselines_{V}.parquet",
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
LIVE_AE_BASELINE_PATH = _local_or(
    "LIVE_AE_BASELINE_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{LV}", "ae_baseline_clean.npy"),
)
LIVE_IF_BASELINE_PATH = _local_or(
    "LIVE_IF_BASELINE_PATH",
    os.path.join(BASE_DIR, "isolation_forests", f"iforest_model_{LV}", "if_baseline_clean.npy"),
)
LIVE_CALIBRATION_THRESHOLD_PATH = _local_or(
    "LIVE_CALIBRATION_THRESHOLD_PATH",
    os.path.join(BASE_DIR, "encoders", f"encoder_model_{LV}", "calibration_thresholds.json"),
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
    ANALYST_TABLE_PARQUET = os.path.join(BASE_DIR, "explainability", "alert_table", f"alert_table_{V}", f"alert_table_{V}.parquet")
    ANALYST_TABLE_CSV = os.path.join(BASE_DIR, "explainability", "alert_table", f"alert_table_{V}", f"alert_table_{V}.csv")

_ueba_override = _local_or("UEBA_PATH", "")
if _ueba_override:
    UEBA_PARQUET = _ueba_override if _ueba_override.endswith(".parquet") else ""
    UEBA_CSV = _ueba_override if _ueba_override.endswith(".csv")     else _ueba_override
else:
    UEBA_PARQUET = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}_train.parquet")
    UEBA_CSV = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}_train.csv")

# UEBA Table A (PC-level drill-down, user/pc/day granularity)
UEBA_A_PARQUET = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}a.parquet")
UEBA_A_CSV = os.path.join(BASE_DIR, "processed_datasets", f"ueba_dataset_{V}", f"ueba_dataset_{V}a.csv")

# Calibration-period alert table (generated by Alert_Object_Builder.ipynb, calibration section)
CALIB_ALERT_TABLE_PARQUET = os.path.join(
    BASE_DIR, "explainability", "alert_table",
    f"alert_table_{V}", f"alert_table_{V}_calib.parquet",
)


# HuggingFace Repository Config
# All repo IDs derive from MODEL_VERSION — update the version, all paths follow.
# Override any of these in paths.local.py for personal forks or staging repos.
HF_ORG              = _local_or("HF_ORG",          "InsiderGuard-AI")
HF_DATASET_REPO     = _local_or("HF_DATASET_REPO", f"{HF_ORG}/ueba-v{V}")
HF_MODEL_REPO       = _local_or("HF_MODEL_REPO",   f"{HF_ORG}/ueba-models-v{V}")
HF_DATASET_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main"
