"""Stage: train-if — train the Isolation Forest on normal-behavior embeddings.

Faithful extraction of Isolation_Forest.ipynb: fit the IF on the AE's
normal-behavior embeddings, score the full training set, build the clean IF
baseline from the insider-free calibration slice, and run the resubstitution +
calibration + SOC metric battery. Metrics are written to metrics_if.json
beside the PNGs (CLEANUP_REPORT gap 6).

Deliberate deviation from the notebook: the band-keyed
calibration_thresholds.json consumed by the live scorer and alert builder is
written ONLY by the `calibrate` stage. (The notebook wrote an incompatible
percentile-keyed file here that the Alert_Object_Builder immediately
overwrote — a latent format conflict this pipeline removes.)
"""

import json
import os

from ueba import config
from ueba.pipeline import manifest
from ueba.pipeline.stages._util import jsonable

STAGE = "train-if"


def requires() -> list[tuple[str, str]]:
    enc = config.SAVE_ENCODER_PATH
    return [
        (os.path.join(enc, "normal_embeddings.npy"), "train-ae"),
        (os.path.join(enc, "full_train_embeddings.npy"), "train-ae"),
        (os.path.join(enc, "train_metadata.csv"), "train-ae"),
        (config.SCALER_PATH, "train-ae"),
        (config.ENCODER_PATH, "train-ae"),
        (config.UEBA_CALIBRATION_PATH, "preprocess"),
        (config.UEBA_CALIB_EVAL_PATH, "preprocess"),
    ]


def produces() -> list[str]:
    save = config.SAVE_IFOREST_PATH
    return [
        os.path.join(save, "iforest_model.pkl"),
        os.path.join(save, "anomaly_scores.npy"),
        os.path.join(save, "anomaly_labels.npy"),
        config.IF_BASELINE_PATH,
        os.path.join(save, "metrics_if.json"),
    ]


def run(args) -> None:
    import joblib
    import numpy as np
    import pandas as pd
    from tensorflow.keras.models import load_model

    from ueba.models.data_prep import build_insider_mask, get_insiders, to_model_matrix
    from ueba.models.isolation_forest import UEBAIsolationForest

    manifest.require(requires())
    save_path = config.SAVE_IFOREST_PATH
    enc_path = config.SAVE_ENCODER_PATH
    os.makedirs(save_path, exist_ok=True)

    normal_embeddings = np.load(os.path.join(enc_path, "normal_embeddings.npy"))
    full_embeddings = np.load(os.path.join(enc_path, "full_train_embeddings.npy"))
    train_metadata = pd.read_csv(os.path.join(enc_path, "train_metadata.csv"))

    contamination = UEBAIsolationForest.compute_contamination_rate(full_embeddings, normal_embeddings)
    print(f"[train-if] Training (contamination={contamination}) ...")
    iforest = UEBAIsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iforest.train(normal_embeddings)

    scores = iforest.anomaly_score(full_embeddings)
    predictions = iforest.predict(full_embeddings)
    print(f"[train-if] Anomalies flagged: {(predictions == -1).sum():,} ({(predictions == -1).mean() * 100:.2f}%)")

    iforest.save(os.path.join(save_path, "iforest_model.pkl"))
    np.save(os.path.join(save_path, "anomaly_scores.npy"), scores)
    np.save(os.path.join(save_path, "anomaly_labels.npy"), predictions)

    print("[train-if] Building clean IF baseline from the insider-free calibration slice ...")
    scaler = joblib.load(config.SCALER_PATH)
    encoder_model = load_model(config.ENCODER_PATH, compile=False)
    calib_clean_df = pd.read_parquet(config.UEBA_CALIBRATION_PATH)
    x_calib_clean, _ = to_model_matrix(calib_clean_df)
    calib_clean_embeddings = encoder_model.predict(scaler.transform(x_calib_clean), batch_size=256)
    baseline_score_dist = iforest.anomaly_score(calib_clean_embeddings)
    np.save(config.IF_BASELINE_PATH, baseline_score_dist)

    # Percentile thresholds used internally by the metric battery below.
    percentile_thresholds = {str(p): float(np.percentile(baseline_score_dist, p)) for p in [80, 90, 95]}

    print("[train-if] Evaluating (resubstitution diagnostics + calibration metrics) ...")
    resub = {
        "separation_ratio": iforest.compute_separation_ratio(full_embeddings, train_metadata["is_insider"]),
        "roc_auc": iforest.compute_roc_auc_score(full_embeddings, train_metadata["is_insider"]),
        "avg_precision": iforest.compute_avg_prec_score(full_embeddings, train_metadata["is_insider"]),
        "recall_at_percentile": iforest.compute_recall_thresholds(full_embeddings, train_metadata["is_insider"]),
    }

    insiders_df = get_insiders(path=config.INSIDERS_PATH, version=config.CERT_VERSION)
    calib_eval_df = pd.read_parquet(config.UEBA_CALIB_EVAL_PATH)
    calib_eval_df["day"] = pd.to_datetime(calib_eval_df["day"]).dt.normalize()
    calib_eval_df["user"] = calib_eval_df["user"].str.strip().str.lower()
    insider_mask_calib = build_insider_mask(calib_eval_df, insiders_df)
    assert insider_mask_calib.sum() > 0, "No insider rows in calibration eval slice"

    x_calib_eval, _ = to_model_matrix(calib_eval_df)
    calib_embeddings = encoder_model.predict(scaler.transform(x_calib_eval), batch_size=256)

    ks_stat, ks_pval = iforest.compute_score_distribution_shift(
        calib_embeddings, baseline_score_dist, percentile_thresholds, save_path=save_path
    )
    calibration = {
        "separation_ratio": iforest.compute_separation_ratio(calib_embeddings, insider_mask_calib),
        "roc_auc": iforest.compute_roc_auc_score(calib_embeddings, insider_mask_calib),
        "avg_precision": iforest.compute_avg_prec_score(calib_embeddings, insider_mask_calib),
        "recall_at_percentile": iforest.compute_recall_thresholds(
            calib_embeddings, insider_mask_calib, threshold_source=baseline_score_dist
        ),
        "confusion_matrix": iforest.compute_confusion_matrix(
            calib_embeddings, insider_mask_calib, baseline_score_dist, save_path=save_path
        ),
        "precision_at_recall": iforest.compute_precision_at_recall(
            calib_embeddings, insider_mask_calib, save_path=save_path
        ),
        "user_detection_rate": iforest.compute_user_detection_rate(
            calib_embeddings, calib_eval_df, insiders_df, baseline_score_dist, save_path=save_path
        ),
        "alert_volume": iforest.compute_alert_volume(
            calib_embeddings, insider_mask_calib, calib_eval_df, baseline_score_dist, save_path=save_path
        ),
        "time_to_first_alert": iforest.compute_time_to_first_alert(
            calib_embeddings, calib_eval_df, insiders_df,
            threshold=percentile_thresholds["90"], save_path=save_path,
        ),
        "score_distribution_shift": {"ks_statistic": ks_stat, "p_value": ks_pval},
    }
    iforest.compute_rank_order(
        calib_embeddings, calib_eval_df,
        insiders_df["user"].str.strip().str.lower().tolist(), save_path=save_path,
    )

    metrics = {
        "contamination": contamination,
        "baseline_percentile_thresholds": percentile_thresholds,
        "resubstitution_diagnostics": resub,
        "calibration": calibration,
    }
    metrics_path = os.path.join(save_path, "metrics_if.json")
    with open(metrics_path, "w") as f:
        json.dump(jsonable(metrics), f, indent=2)
    print(f"[train-if] Metrics written to {metrics_path}")

    manifest.record(STAGE, produces())
