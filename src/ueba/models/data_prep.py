# Imports
#
# Heavy model dependencies (joblib, tensorflow, the IF wrapper) are imported
# lazily inside get_scores(): every other function here is pure pandas/numpy,
# and the unit suite must be able to import this module in a tensorflow-free
# environment (CI).
import numpy as np
import pandas as pd

from ueba import config
from ueba.features.preprocessing import chronological_split  # re-export for backward compatibility

__all__ = ["chronological_split", "get_insiders", "build_insider_mask", "prepare_ae_training_data", "to_model_matrix", "get_scores"]


SCALER_PATH = config.SCALER_PATH
ENCODER_PATH = config.ENCODER_PATH
IF_PATH = config.IF_PATH


def get_insiders(path: str, version: str | float, return_all: bool=False) -> pd.DataFrame:
    """
    Loads the raw insiders table and returns the scenarios for a specific version. If specified, the
    full raw table across all CERT versions can be returned.

    Args:
        path: The location of the insiders CSV file
        version: CERT version to extract
        return_all: If true, returns the full insider table along with the version-specific table

    Returns:
        pd.DataFrame: A table describing the window of abnormal activity for each insider
    """
    df = pd.read_csv(path)

    # Normalizing table
    df.columns = df.columns.str.strip()
    df["user"] = df["user"].str.strip().str.lower()
    df["start"] = pd.to_datetime(df["start"].str.strip())
    df["end"] = pd.to_datetime(df["end"].str.strip())
    df["start_day"] = df["start"].dt.normalize()
    df["end_day"] = df["end"].dt.normalize()

    if isinstance(version, float):
        version = str(version)

    # Extracting only version-specific insiders
    insiders_df = df[df["dataset"].astype(str).str.strip() == version]
    insiders_df = insiders_df[["user", "start_day", "end_day", "scenario"]].reset_index(drop=True)

    if return_all:
        return insiders_df, df
    else:
        return insiders_df


def build_insider_mask(df: pd.DataFrame, windows: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean mask for any (user, day) pair that falls within a known threat window.

    Args:
        df: The UEBA dataset intended for model training
        windows: DataFrame holding the insider windows

    Returns:
        pd.Series: A boolean mask where `False=normal` and `True=insider`
    """
    mask = pd.Series(False, index=df.index)
    for _, row in windows.iterrows():
        mask |= (
            (df["user"] == row["user"]) &
            (df["day"] >= row["start_day"]) &
            (df["day"] <= row["end_day"])
        )
    return mask


def prepare_ae_training_data(
    train_df: pd.DataFrame,
    insiders_df: pd.DataFrame,
    val_ratio: float=0.15,
) -> tuple:
    """
    Prepares clean training splits for Autoencoder training.

    Applies baseline gating, separates insider rows, and produces a
    chronological fit/validation split on normal-behavior data only.

    Args:
        train_df: The calibration-excluded training set (output of CERT_Preprocessing).
        insiders_df: Ground-truth insider windows from `get_insiders()`.
        val_ratio: Fraction of normal-behavior rows held out for validation (default 0.15).

    Returns:
        tuple: [train_fit, train_val, train_normal, train_df, insider_mask]
    """
    # Ensuring there's at least 14 days of prior history
    train_df = train_df[train_df["baseline_complete"]].copy().reset_index(drop=True)

    # Building insider mask
    insider_mask = build_insider_mask(train_df, insiders_df)
    train_normal = train_df[~insider_mask].copy()

    # Creating validation set
    train_fit, train_val = chronological_split(df=train_normal, split_ratio=1.0-val_ratio)

    return (train_fit, train_val, train_normal, train_df, insider_mask)


def to_model_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Selects the numeric behavioral feature matrix for model training/inference.

    Drops the non-feature columns defined in ``config.NON_FEATURE_COLS``
    (identifiers, LDAP profile/identity enrichment, and gating flags) and
    returns a float32 matrix. Asserts that every remaining column is numeric so
    that a newly added string column (e.g. a future LDAP attribute) fails loudly
    here instead of as an opaque ``could not convert string to float`` later.

    Args:
        df: A UEBA (user, day) or (user, pc, day) DataFrame.

    Returns:
        tuple: (float32 feature matrix, ordered list of feature column names).
    """
    X = df.drop(columns=[c for c in config.NON_FEATURE_COLS if c in df.columns])
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    assert not non_numeric, f"Non-numeric columns leaked into model matrix: {non_numeric}"
    return X.values.astype("float32"), X.columns.tolist()


def get_scores(newData_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a raw data and returns a DataFrame with a corresponding anomaly score.

    Args:
        newData_df: The raw DataFrame

    Returns:
        pd.DataFrame: DataFrame with corresponding anomaly score
    """
    import joblib
    from tensorflow.keras.models import load_model

    from ueba.models.isolation_forest import UEBAIsolationForest

    # Use the parameter, work on a copy
    live_df = newData_df.copy()

    # remember user IDs if present
    user_ids = live_df["user"] if "user" in live_df.columns else None

    # Select the numeric behavioral feature matrix (drops identifiers, LDAP
    # profile enrichment, and gating flags via config.NON_FEATURE_COLS)
    x_live, _ = to_model_matrix(live_df)

    # Load pre-fitted scaler
    scaler = joblib.load(SCALER_PATH)
    live_scaled = scaler.transform(x_live)

    # Load encoder and generate embeddings in batch
    encoder = load_model(ENCODER_PATH)
    embeddings = encoder.predict(live_scaled)

    # Load IF and generate anomaly scores
    iforest = UEBAIsolationForest()
    iforest.load(IF_PATH)
    scores = iforest.anomaly_score(embeddings)

    # build a results dataframe
    result_df = pd.DataFrame({"anomaly_score": scores})
    if user_ids is not None:
        # insert user column at front
        result_df.insert(0, "user", user_ids.values)
    # align index with input
    result_df.index = newData_df.index

    return result_df


