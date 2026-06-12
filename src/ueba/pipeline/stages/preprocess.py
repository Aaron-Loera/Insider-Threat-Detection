"""Stage: preprocess — raw CERT logs -> v{V} feature datasets.

Faithful extraction of CERT_Preprocessing.ipynb: Layer A (user, pc, day) with
per-user work-hour envelopes, Layer B (user, day) with the full v6 feature
set, the chronological 80/10/10 train/calibration/test_stream splits, and the
insider-free calibration variant. Also generates peer_baselines_{V}.parquet
(department x day feature means) for the dashboard's Investigation tab —
previously produced ad hoc and absent from the repo.
"""

import os

from ueba import config
from ueba.pipeline import manifest

STAGE = "preprocess"


def requires() -> list[tuple[str, str]]:
    if not config.CERT_PATH:
        raise manifest.MissingArtifactError(
            "CERT_PATH is not set. Point it at the raw CERT dataset in paths.local.py."
        )
    return [
        (os.path.join(config.CERT_PATH, "logon.csv"), "external: raw CERT dataset"),
        (config.INSIDERS_PATH, "external: CERT answers/insiders.csv"),
    ]


def produces() -> list[str]:
    return [
        config.UEBA_A_PATH,
        config.UEBA_B_PATH,
        config.UEBA_PATH,
        config.UEBA_CALIBRATION_PATH,
        config.UEBA_CALIB_EVAL_PATH,
        config.TEST_STREAM_PATH,
        config.USER_WORK_HOURS_PATH,
        config.PEER_BASELINES_PATH,
    ]


def _build_peer_baselines(train_df) -> "pd.DataFrame":  # noqa: F821 — pandas imported in run()
    """(department, day, <feature> means) table for peer-comparison charts."""
    import pandas as pd

    numeric_cols = train_df.select_dtypes(include="number").columns
    feature_cols = [c for c in numeric_cols if c not in config.NON_FEATURE_COLS]
    out = (
        train_df.groupby(["department", "day"], observed=True)[feature_cols]
        .mean()
        .reset_index()
    )
    out["day"] = pd.to_datetime(out["day"]).dt.normalize()
    return out


def run(args) -> None:

    from ueba.features.preprocessing import (
        build_layer_a,
        build_layer_b,
        chronological_split,
        load_ldap,
        save_dataset,
        save_nunique_frames,
    )
    from ueba.models.data_prep import get_insiders

    manifest.require(requires())
    mv = config.MODEL_VERSION
    out_dir = config.DATASET_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f"[preprocess] Building Layer A from {config.CERT_PATH} ...")
    layer_a_dataset, nunique_frames = build_layer_a(
        cert_path=config.CERT_PATH,
        work_hours=(9, 17),
        return_nunique_frames=True,
        compute_schedules=True,
        save_schedule_to=config.USER_WORK_HOURS_PATH,
    )
    save_dataset(layer_a_dataset, f"ueba_dataset_{mv}a.parquet", out_dir)
    save_nunique_frames(nunique_frames, config.SAFEPOINT_DIR)

    print("[preprocess] Building Layer B ...")
    ldap_df = load_ldap(config.CERT_PATH)
    layer_b_dataset = build_layer_b(
        layer_a_df=layer_a_dataset,
        rolling_window=5,
        nunique_frames=nunique_frames,
        ldap_df=ldap_df,
        peer_col=config.PEER_GROUP_KEY,
    )
    save_dataset(layer_b_dataset, f"ueba_dataset_{mv}b.parquet", out_dir)

    print("[preprocess] Creating chronological splits ...")
    train_df, test_df = chronological_split(df=layer_b_dataset, split_ratio=0.9)
    test_df.to_parquet(config.TEST_STREAM_PATH, index=False)

    train_df, calib_df = chronological_split(df=train_df, split_ratio=8 / 9)
    insiders_df = get_insiders(path=config.INSIDERS_PATH, version=config.CERT_VERSION)
    insider_ids = set(insiders_df["user"].unique())
    calib_clean_df = calib_df[~calib_df["user"].isin(insider_ids)].copy()

    calib_clean_df.to_parquet(config.UEBA_CALIBRATION_PATH, index=False)
    calib_df.to_parquet(config.UEBA_CALIB_EVAL_PATH, index=False)
    train_df.to_parquet(config.UEBA_PATH, index=False)

    print("[preprocess] Building peer baselines ...")
    peer = _build_peer_baselines(train_df)
    peer.to_parquet(config.PEER_BASELINES_PATH, index=False)

    print(
        f"[preprocess] Train: {len(train_df):,}  Calibration(clean): {len(calib_clean_df):,}  "
        f"CalibrationEval: {len(calib_df):,}  TestStream: {len(test_df):,}"
    )
    manifest.record(STAGE, produces())
