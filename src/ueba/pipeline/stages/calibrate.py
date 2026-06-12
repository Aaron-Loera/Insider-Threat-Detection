"""Stage: calibrate — clean baselines + band-keyed absolute thresholds.

Faithful extraction of the "Building Clean Calibration Baseline and
Calibrating Absolute Thresholds" section of Alert_Object_Builder.ipynb. This
is the sole producer of calibration_thresholds.json in the canonical
band-keyed format ({"ae": {LOW..CRITICAL}, "if": {...}}) consumed by both
the live scorer and the alert builder. (Historically the IF notebook wrote a
percentile-keyed file to the same path that the builder then overwrote.)

By default baselines are recomputed from the models (notebook-faithful).
--thresholds-only derives the JSON from the existing baseline .npy files in
seconds — the recovery path for the missing-thresholds condition found
during the cleanup audit.
"""

import json
import os

from ueba import config
from ueba.pipeline import manifest

STAGE = "calibrate"

# Target alert budget on clean user-days: ~5% exceed LOW, ~1% MEDIUM,
# ~0.5% HIGH; CRITICAL has no upper bound.
BAND_PERCENTILES = {"LOW": 95.0, "MEDIUM": 99.0, "HIGH": 99.5}


def requires(thresholds_only: bool = False) -> list[tuple[str, str]]:
    if thresholds_only:
        return [
            (config.AE_BASELINE_PATH, "train-ae"),
            (config.IF_BASELINE_PATH, "train-if"),
        ]
    return [
        (config.UEBA_CALIBRATION_PATH, "preprocess"),
        (config.SCALER_PATH, "train-ae"),
        (config.AE_PATH, "train-ae"),
        (config.ENCODER_PATH, "train-ae"),
        (config.IF_PATH, "train-if"),
        (config.INSIDERS_PATH, "external: CERT answers/insiders.csv"),
    ]


def produces() -> list[str]:
    return [
        config.AE_BASELINE_PATH,
        config.IF_BASELINE_PATH,
        config.CALIBRATION_THRESHOLD_PATH,
    ]


def _band_thresholds(values) -> dict:
    import numpy as np

    out = {band: float(np.percentile(values, pct)) for band, pct in BAND_PERCENTILES.items()}
    out["CRITICAL"] = None  # no upper bound; converted to inf on load
    return out


def run(args) -> None:
    import numpy as np

    manifest.require(requires(thresholds_only=args.thresholds_only))

    if args.thresholds_only:
        print("[calibrate] Deriving thresholds from existing baseline arrays ...")
        ae_calib_errors = np.load(config.AE_BASELINE_PATH)
        if_calib_scores = np.load(config.IF_BASELINE_PATH)
    else:
        import joblib
        import pandas as pd
        from tensorflow.keras.models import load_model

        from ueba.models.data_prep import get_insiders, to_model_matrix
        from ueba.models.isolation_forest import UEBAIsolationForest

        print("[calibrate] Scoring the insider-free calibration slice through AE + IF ...")
        calib_df = pd.read_parquet(config.UEBA_CALIBRATION_PATH)
        calib_df["day"] = pd.to_datetime(calib_df["day"]).dt.normalize()
        calib_df["user"] = calib_df["user"].str.strip().str.lower()

        insiders_df = get_insiders(path=config.INSIDERS_PATH, version=config.CERT_VERSION)
        insider_users = set(insiders_df["user"].str.strip().str.lower().unique())
        calib_clean = calib_df[~calib_df["user"].isin(insider_users)].copy()
        print(f"[calibrate] Calibration rows after insider exclusion: {len(calib_clean):,} / {len(calib_df):,}")

        scaler = joblib.load(config.SCALER_PATH)
        ae_model = load_model(config.AE_PATH, compile=False)
        enc_model = load_model(config.ENCODER_PATH, compile=False)
        iforest = UEBAIsolationForest()
        iforest.load(config.IF_PATH)

        x_calib, _ = to_model_matrix(calib_clean)
        x_calib_scaled = scaler.transform(x_calib)

        ae_reconstructed = ae_model.predict(x_calib_scaled, batch_size=4096)
        ae_calib_errors = np.sum(np.square(x_calib_scaled - ae_reconstructed.astype("float32")), axis=1)
        calib_embeddings = enc_model.predict(x_calib_scaled, batch_size=4096)
        if_calib_scores = iforest.anomaly_score(calib_embeddings)

        os.makedirs(os.path.dirname(config.AE_BASELINE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(config.IF_BASELINE_PATH), exist_ok=True)
        np.save(config.AE_BASELINE_PATH, ae_calib_errors)
        np.save(config.IF_BASELINE_PATH, if_calib_scores)
        print(f"[calibrate] AE baseline: {len(ae_calib_errors):,} values  IF baseline: {len(if_calib_scores):,} values")

    thresholds = {"ae": _band_thresholds(ae_calib_errors), "if": _band_thresholds(if_calib_scores)}
    os.makedirs(os.path.dirname(config.CALIBRATION_THRESHOLD_PATH), exist_ok=True)
    with open(config.CALIBRATION_THRESHOLD_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)

    ae_t, if_t = thresholds["ae"], thresholds["if"]
    print(f"[calibrate] AE thresholds — LOW: {ae_t['LOW']:.2f}  MEDIUM: {ae_t['MEDIUM']:.2f}  HIGH: {ae_t['HIGH']:.2f}  CRITICAL: inf")
    print(f"[calibrate] IF thresholds — LOW: {if_t['LOW']:.4f}  MEDIUM: {if_t['MEDIUM']:.4f}  HIGH: {if_t['HIGH']:.4f}  CRITICAL: inf")
    print(f"[calibrate] Saved {config.CALIBRATION_THRESHOLD_PATH}")

    manifest.record(STAGE, produces())
