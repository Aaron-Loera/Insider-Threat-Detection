"""Stage: explain — AE reconstruction-error tables for a chosen split.

Faithful extraction of Reconstruction_Error_Explainer.ipynb, parametrized
over splits so the test-stream table (historically never generated — the
root cause of CLEANUP_REPORT gap 3) is one command:

    python -m ueba.pipeline explain --split train
    python -m ueba.pipeline explain --split test_stream
"""

import os

from ueba import config
from ueba.pipeline import manifest

STAGE = "explain"

_SPLIT_INPUT = {
    "train": lambda: config.UEBA_PATH,
    "test_stream": lambda: config.TEST_STREAM_PATH,
}
_SPLIT_OUTPUT = {
    "train": lambda: config.RECON_TABLE_PATH,
    "test_stream": lambda: config.RECON_TEST_TABLE_PATH,
}


def requires(split: str = "train") -> list[tuple[str, str]]:
    return [
        (_SPLIT_INPUT[split](), "preprocess"),
        (config.SCALER_PATH, "train-ae"),
        (config.AE_PATH, "train-ae"),
    ]


def produces(split: str = "train") -> list[str]:
    return [_SPLIT_OUTPUT[split]()]


def run(args) -> None:
    import joblib
    from tensorflow.keras.models import load_model

    from ueba.alerts.explainer import ReconstructionErrorExplainer, build_feature_groups
    from ueba.models.data_prep import to_model_matrix
    from ueba.pipeline.stages._util import load_split_frame

    split = args.split
    manifest.require(requires(split))
    in_path = _SPLIT_INPUT[split]()
    out_path = _SPLIT_OUTPUT[split]()

    print(f"[explain] Loading {split} split: {in_path}")
    df = load_split_frame(in_path)
    df = df.sort_values("day").reset_index(drop=True)
    print(f"[explain] {len(df):,} rows after baseline_complete gate")

    scaler = joblib.load(config.SCALER_PATH)
    matrix, feature_names = to_model_matrix(df)
    scaled = scaler.transform(matrix)

    print("[explain] Loading autoencoder and decomposing reconstruction error ...")
    ae = load_model(config.AE_PATH, compile=False)
    explainer = ReconstructionErrorExplainer(
        feature_names=feature_names,
        feature_groups=build_feature_groups(feature_names),
    )
    recon_table = explainer.explain_to_df(
        scaled, ae,
        metadata=df[["user", "day"]],
        include_feat_err=False,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = recon_table.copy()
    float_cols = out.select_dtypes(include="float64").columns
    if len(float_cols):
        out[float_cols] = out[float_cols].astype("float32")
    out.to_parquet(out_path, index=False)
    print(f"[explain] {len(out):,} rows -> {out_path}")

    manifest.record(STAGE, produces(split))
