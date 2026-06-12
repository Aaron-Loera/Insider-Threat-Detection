"""Stage: build-alerts — SOC-ready alert tables + aggregated cases.

Faithful extraction of Alert_Object_Builder.ipynb, parametrized over the
three scored populations:

    --split main   train-period table from the precomputed reconstruction
                   table + anomaly scores (alert_table_{V}.parquet, cases)
    --split calib  calibration-eval period, recon errors + IF scores computed
                   inline (alert_table_{V}_calib.parquet, cases)
    --split test   held-out test stream; REQUIRES the test reconstruction
                   table from `explain --split test_stream` and fails loudly
                   if it is absent instead of silently skipping
                   (CLEANUP_REPORT gap 3); IF scores computed inline and also
                   persisted as test_anomaly_scores.npy for the notebook path

All splits band through the calibrated thresholds + clean baselines produced
by the `calibrate` stage.
"""

import json

from ueba import config
from ueba.pipeline import manifest

STAGE = "build-alerts"


def requires(split: str = "main") -> list[tuple[str, str]]:
    common = [
        (config.CALIBRATION_THRESHOLD_PATH, "calibrate"),
        (config.AE_BASELINE_PATH, "calibrate"),
        (config.IF_BASELINE_PATH, "calibrate"),
    ]
    if split == "main":
        return common + [
            (config.RECON_TABLE_PATH, "explain --split train"),
            (config.IF_SCORES_PATH, "train-if"),
            (config.UEBA_PATH, "preprocess"),
        ]
    if split == "calib":
        return common + [
            (config.UEBA_CALIB_EVAL_PATH, "preprocess"),
            (config.SCALER_PATH, "train-ae"),
            (config.AE_PATH, "train-ae"),
            (config.ENCODER_PATH, "train-ae"),
            (config.IF_PATH, "train-if"),
        ]
    if split == "test":
        return common + [
            (config.RECON_TEST_TABLE_PATH, "explain --split test_stream"),
            (config.TEST_STREAM_PATH, "preprocess"),
            (config.SCALER_PATH, "train-ae"),
            (config.ENCODER_PATH, "train-ae"),
            (config.IF_PATH, "train-if"),
        ]
    raise ValueError(f"Unknown split: {split}")


def produces(split: str = "main") -> list[str]:
    if split == "main":
        return [config.ANALYST_TABLE_PARQUET, config.CASES_PARQUET]
    if split == "calib":
        return [config.CALIB_ALERT_TABLE_PARQUET, config.CALIB_CASES_PARQUET]
    if split == "test":
        return [config.TEST_ALERT_TABLE_PARQUET, config.TEST_CASES_PARQUET, config.TEST_IF_SCORES_PATH]
    raise ValueError(f"Unknown split: {split}")


def _calibrated_builder():
    """AlertObjectBuilder armed with the calibrated thresholds + clean baselines."""
    import numpy as np

    from ueba.alerts.builder import AlertObjectBuilder

    with open(config.CALIBRATION_THRESHOLD_PATH) as f:
        thresholds = json.load(f)
    builder = AlertObjectBuilder(
        top_k=3,
        ae_absolute_thresholds=thresholds["ae"],
        if_absolute_thresholds=thresholds["if"],
    )
    builder.fit_ae_baseline(np.load(config.AE_BASELINE_PATH))
    builder.fit_if_baseline(np.load(config.IF_BASELINE_PATH))
    return builder


def _context_table(df):
    """(user, day) + z-score/rolling-delta columns used for explanation text."""
    zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
    delta_cols = [c for c in df.columns if c.endswith("_rolling_delta")]
    return df[["user", "day"] + zscore_cols + delta_cols].copy()


def _merge_aligned(recon_table, score_table, what: str):
    merged = recon_table.merge(score_table, on=["user", "day"], how="inner")
    assert len(merged) == len(recon_table) == len(score_table), (
        f"{what}: recon and score tables must align after the baseline_complete filter. "
        f"Got recon={len(recon_table):,}, score={len(score_table):,}, merged={len(merged):,}"
    )
    return merged


def run(args) -> None:
    import numpy as np
    import pandas as pd

    from ueba.alerts.builder import save_table
    from ueba.pipeline.stages._util import load_split_frame

    split = args.split
    manifest.require(requires(split), allow_missing=args.allow_missing)
    builder = _calibrated_builder()

    if split == "main":
        print("[build-alerts] Loading train reconstruction table + anomaly scores ...")
        recon_table = pd.read_parquet(config.RECON_TABLE_PATH)
        ueba_dataset = load_split_frame(config.UEBA_PATH)
        score_table = _context_table(ueba_dataset)
        score_table["if_anomaly_score"] = np.load(config.IF_SCORES_PATH)
        aggregated = _merge_aligned(recon_table, score_table, "train")

        alert_df = builder.build_alert_df(aggregated, w1=0.5, w2=0.5)
        save_table(alert_df, config.ANALYST_TABLE_PARQUET)
        cases_df = builder.aggregate_alerts(alert_df, window_days=5, min_risk="HIGH")
        save_table(cases_df, config.CASES_PARQUET)

    elif split == "calib":
        import joblib
        from tensorflow.keras.models import load_model

        from ueba.alerts.explainer import ReconstructionErrorExplainer
        from ueba.models.data_prep import to_model_matrix
        from ueba.models.isolation_forest import UEBAIsolationForest

        print("[build-alerts] Scoring the calibration-eval slice inline ...")
        calib_eval_df = load_split_frame(config.UEBA_CALIB_EVAL_PATH)
        x, feature_cols = to_model_matrix(calib_eval_df)
        scaler = joblib.load(config.SCALER_PATH)
        x_scaled = scaler.transform(x)

        ae_model = load_model(config.AE_PATH, compile=False)
        recon_table = ReconstructionErrorExplainer(feature_names=feature_cols).explain_to_df(
            x_scaled, ae_model,
            metadata=calib_eval_df[["user", "day"]],
            include_feat_err=False,
            include_contributions=True,
        )

        enc_model = load_model(config.ENCODER_PATH, compile=False)
        iforest = UEBAIsolationForest()
        iforest.load(config.IF_PATH)
        score_table = _context_table(calib_eval_df)
        score_table["if_anomaly_score"] = iforest.anomaly_score(
            enc_model.predict(x_scaled, batch_size=4096)
        )
        aggregated = _merge_aligned(recon_table, score_table, "calibration")

        alert_df = builder.build_alert_df(aggregated, w1=0.5, w2=0.5)
        save_table(alert_df, config.CALIB_ALERT_TABLE_PARQUET)
        cases_df = builder.aggregate_alerts(alert_df, window_days=5, min_risk="HIGH")
        save_table(cases_df, config.CALIB_CASES_PARQUET)

    elif split == "test":
        import joblib
        from tensorflow.keras.models import load_model

        from ueba.models.data_prep import to_model_matrix
        from ueba.models.isolation_forest import UEBAIsolationForest

        print("[build-alerts] Loading test reconstruction table; scoring test stream through IF ...")
        recon_table = pd.read_parquet(config.RECON_TEST_TABLE_PATH)
        test_df = load_split_frame(config.TEST_STREAM_PATH)
        x, _ = to_model_matrix(test_df)
        scaler = joblib.load(config.SCALER_PATH)
        x_scaled = scaler.transform(x)

        enc_model = load_model(config.ENCODER_PATH, compile=False)
        iforest = UEBAIsolationForest()
        iforest.load(config.IF_PATH)
        test_scores = iforest.anomaly_score(enc_model.predict(x_scaled, batch_size=4096))
        np.save(config.TEST_IF_SCORES_PATH, test_scores)

        score_table = _context_table(test_df)
        score_table["if_anomaly_score"] = test_scores
        aggregated = _merge_aligned(recon_table, score_table, "test")

        alert_df = builder.build_alert_df(aggregated, w1=0.5, w2=0.5)
        save_table(alert_df, config.TEST_ALERT_TABLE_PARQUET)
        cases_df = builder.aggregate_alerts(alert_df, window_days=5, min_risk="HIGH")
        save_table(cases_df, config.TEST_CASES_PARQUET)

    else:
        raise ValueError(f"Unknown split: {split}")

    print(f"[build-alerts] {split}: {len(alert_df):,} alerts, {len(cases_df):,} cases")
    print(alert_df["ae_risk_band"].value_counts().to_string())
    manifest.record(STAGE, produces(split))
