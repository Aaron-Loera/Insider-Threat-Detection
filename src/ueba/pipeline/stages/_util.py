"""Shared helpers for pipeline stages."""

import numpy as np
import pandas as pd


def jsonable(obj):
    """Coerce numpy/pandas containers into JSON-serializable structures."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        return jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return jsonable(obj.to_dict())
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def load_split_frame(path: str) -> pd.DataFrame:
    """Load a (user, day) split parquet/csv with the pipeline's normalization:
    day normalized to midnight, user stripped+lowercased, baseline_complete
    gate applied (mirrors the notebooks' alignment requirements)."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0)
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    df["user"] = df["user"].str.strip().str.lower()
    df = df[df["baseline_complete"]].reset_index(drop=True)
    return df
