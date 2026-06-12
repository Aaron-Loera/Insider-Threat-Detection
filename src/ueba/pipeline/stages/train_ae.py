"""Stage: train-ae — train the autoencoder on insider-filtered normal behavior.

Faithful extraction of Autoencoder.ipynb: prepare insider-free chronological
fit/validation splits, fit the StandardScaler, train the AE (16-dim latent,
hidden 256/128/64, early stopping), persist model + scaler + feature-column
contract + embeddings + the clean AE baseline, and evaluate on the held-out
calibration eval slice. Evaluation metrics are additionally written as
machine-readable metrics_ae.json beside the PNGs (CLEANUP_REPORT gap 6).
"""

import json
import os

from ueba import config
from ueba.pipeline import manifest
from ueba.pipeline.stages._util import jsonable

STAGE = "train-ae"


def requires() -> list[tuple[str, str]]:
    return [
        (config.UEBA_PATH, "preprocess"),
        (config.UEBA_CALIBRATION_PATH, "preprocess"),
        (config.UEBA_CALIB_EVAL_PATH, "preprocess"),
        (config.INSIDERS_PATH, "external: CERT answers/insiders.csv"),
    ]


def produces() -> list[str]:
    save = config.SAVE_ENCODER_PATH
    return [
        os.path.join(save, "autoencoder_model.keras"),
        os.path.join(save, "encoder_model.keras"),
        os.path.join(save, "feature_scaler.pkl"),
        os.path.join(save, "feature_cols.json"),
        os.path.join(save, "normal_embeddings.npy"),
        os.path.join(save, "full_train_embeddings.npy"),
        os.path.join(save, "train_metadata.csv"),
        config.AE_BASELINE_PATH,
        os.path.join(save, "metrics_ae.json"),
    ]


def run(args) -> None:
    import joblib
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    from ueba.models.autoencoder import Autoencoder, plot_loss
    from ueba.models.data_prep import (
        build_insider_mask,
        get_insiders,
        prepare_ae_training_data,
        to_model_matrix,
    )

    manifest.require(requires())
    save_path = config.SAVE_ENCODER_PATH
    os.makedirs(save_path, exist_ok=True)
    tf.random.set_seed(42)

    print("[train-ae] Preparing training splits ...")
    train_df = pd.read_parquet(config.UEBA_PATH)
    insiders_df = get_insiders(path=config.INSIDERS_PATH, version=config.CERT_VERSION)
    train_fit, train_val, train_normal, train_df, insider_mask = prepare_ae_training_data(
        train_df=train_df, insiders_df=insiders_df
    )

    x_train_fit, feature_cols = to_model_matrix(train_fit)
    x_train_val, _ = to_model_matrix(train_val)
    x_train, _ = to_model_matrix(train_df)

    scaler = StandardScaler()
    x_train_fit_scaled = scaler.fit_transform(x_train_fit)
    x_train_val_scaled = scaler.transform(x_train_val)
    x_train_scaled = scaler.transform(x_train)
    for name, arr in [("fit", x_train_fit_scaled), ("val", x_train_val_scaled), ("full", x_train_scaled)]:
        assert np.isfinite(arr).all(), f"Non-finite values in scaled {name} matrix. Check z-score clipping."

    print(f"[train-ae] Training on {len(x_train_fit_scaled):,} rows x {x_train_fit_scaled.shape[1]} features ...")
    ae = Autoencoder(
        input_dim=x_train_fit_scaled.shape[1],
        latent_dim=16,
        hidden_dims=(256, 128, 64),
        learning_rate=1e-3,
    )
    history = ae.train(
        x_train=x_train_fit_scaled,
        save_path=save_path,
        epochs=args.epochs,
        batch_size=256,
        x_val=x_train_val_scaled,
    )
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_val_loss = float(min(history.history["val_loss"]))
    print(f"[train-ae] Best validation loss (epoch {best_epoch}): {best_val_loss:.4f}")
    plot_loss(history, save_path)

    print("[train-ae] Persisting model, scaler, contract, embeddings ...")
    ae.autoencoder.save(os.path.join(save_path, "autoencoder_model.keras"))
    ae.encoder.save(os.path.join(save_path, "encoder_model.keras"))
    joblib.dump(scaler, os.path.join(save_path, "feature_scaler.pkl"))
    with open(os.path.join(save_path, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    x_train_normal, _ = to_model_matrix(train_normal)
    normal_embeddings = ae.encode(scaler.transform(x_train_normal))
    full_train_embeddings = ae.encode(x_train_scaled)
    np.save(os.path.join(save_path, "normal_embeddings.npy"), normal_embeddings)
    np.save(os.path.join(save_path, "full_train_embeddings.npy"), full_train_embeddings)

    metadata_df = train_df[["user", "day"]].reset_index(drop=True)
    metadata_df["is_insider"] = insider_mask.values
    metadata_df.to_csv(os.path.join(save_path, "train_metadata.csv"), index=False)

    print("[train-ae] Building clean AE baseline from the insider-free calibration slice ...")
    calib_clean_df = pd.read_parquet(config.UEBA_CALIBRATION_PATH)
    x_calib_clean, _ = to_model_matrix(calib_clean_df)
    ae_baseline_errors = ae.reconstruction_error(scaler.transform(x_calib_clean))
    np.save(config.AE_BASELINE_PATH, ae_baseline_errors)

    print("[train-ae] Evaluating on the held-out calibration eval slice ...")
    calib_eval_df = pd.read_parquet(config.UEBA_CALIB_EVAL_PATH)
    calib_eval_df["day"] = pd.to_datetime(calib_eval_df["day"]).dt.normalize()
    calib_eval_df["user"] = calib_eval_df["user"].str.strip().str.lower()
    insider_mask_calib = build_insider_mask(calib_eval_df, insiders_df)
    assert insider_mask_calib.sum() > 0, "No insider rows in calibration eval slice"

    x_calib_eval, feature_cols_calib = to_model_matrix(calib_eval_df)
    x_calib_eval_scaled = scaler.transform(x_calib_eval)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "roc_auc": ae.compute_roc_auc_score(x_calib_eval_scaled, insider_mask_calib, save_path=save_path),
        "avg_precision": ae.compute_avg_prec_score(x_calib_eval_scaled, insider_mask_calib, save_path=save_path),
        "recall_at_percentile": ae.compute_recall_thresholds(
            x_calib_eval_scaled, insider_mask_calib,
            threshold_source=ae_baseline_errors, save_path=save_path,
        ),
        "channel_reconstruction_error": ae.compute_channel_reconstruction_error(
            x_calib_eval_scaled, insider_mask_calib,
            feature_names=feature_cols_calib, save_path=save_path,
        ),
    }
    metrics_path = os.path.join(save_path, "metrics_ae.json")
    with open(metrics_path, "w") as f:
        json.dump(jsonable(metrics), f, indent=2)
    print(f"[train-ae] Metrics written to {metrics_path}")

    manifest.record(STAGE, produces())
