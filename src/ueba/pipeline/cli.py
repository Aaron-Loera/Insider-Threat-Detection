"""
python -m ueba.pipeline — headless, reproducible ML pipeline.

Stages (in dependency order):

    preprocess        raw CERT logs -> v{V} feature datasets + splits
    train-ae          autoencoder + scaler + feature contract + clean AE baseline
    train-if          isolation forest + anomaly scores + clean IF baseline
    explain           reconstruction-error table  (--split train|test_stream)
    calibrate         clean baselines + band-keyed calibration_thresholds.json
    build-alerts      alert tables + cases        (--split main|calib|test)
    build-dashboard   slim serving parquet for Streamlit Cloud
    all               run everything above in order, stop at first failure
    status            audit the artifact tree + recorded manifest

Every stage validates its inputs through the manifest: a missing artifact is
a hard error naming the producing stage, never a silent skip. `--version N`
re-points every derived path (sets UEBA_MODEL_VERSION before config loads).
"""
import argparse
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI argument parser for the UEBA pipeline.
    
    Sets up a command structure with one global "--version" flag, a required stage
    subcommand ("preprocess", "train-ae", "train-if", etc.) and stage-specific
    options (e.g., "--epochs" for "train-ae", "--split" for "explain"). Returns the
    fully wired parser, ready for `parse_args()` in `main()`.
    
    Args:
        None:
        
    Returns:
        argparse.ArgumentParser: Configured parser for the pipeline CLI.
    """
    # Create root parser
    parser = argparse.ArgumentParser(
        prog="python -m ueba.pipeline",
        description="UEBA insider-threat detection pipeline",
    )

    # Register global "--version" flag
    parser.add_argument("--version", default=None, help="model/dataset version (default: config.MODEL_VERSION)")

    # Create subparsers
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("preprocess", help="raw CERT logs -> feature datasets and splits")

    p = sub.add_parser("train-ae", help="train the autoencoder")
    p.add_argument("--epochs", type=int, default=100)

    sub.add_parser("train-if", help="train the isolation forest")

    p = sub.add_parser("explain", help="build a reconstruction-error table")
    p.add_argument("--split", choices=["train", "test_stream"], default="train")

    p = sub.add_parser("calibrate", help="clean baselines and absolute band thresholds")
    p.add_argument(
        "--thresholds-only",
        action="store_true",
        help="derive thresholds from the existing baseline .npy files without re-scoring"
    )

    p = sub.add_parser("build-alerts", help="build alert tables and cases")
    p.add_argument("--split", choices=["main", "calib", "test"], default="main")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="escape hatch: proceed despite missing inputs (the historical silent-skip behavior, made explicit)"
    )

    sub.add_parser("build-dashboard", help="build the slim dashboard serving parquet")
    sub.add_parser("all", help="run the full pipeline in order")
    sub.add_parser("status", help="audit artifacts + recorded manifest")
    return parser


def _stage_modules() -> dict:
    """
    Load and index all pipeline stage modules by their CLI command name.
    
    Imports are performed lazily (inside the function) so the `UEBA_MODEL_VERSION` can
    be set in the environment before `ueba.config` loads its defaults. Returns a
    dictionary for quick lookup.
    
    Args:
        None:
        
    Returns:
        dict: Dictionary mapping stage names to stage module objects.
    """
    # Lazily import so "UEBA_MODEL_VERSION" is set before ueba.config loads
    from ueba.pipeline.stages import (
        build_alerts,
        build_dashboard,
        calibrate,
        explain,
        preprocess,
        train_ae,
        train_if,
    )

    return {
        "preprocess": preprocess,
        "train-ae": train_ae,
        "train-if": train_if,
        "explain": explain,
        "calibrate": calibrate,
        "build-alerts": build_alerts,
        "build-dashboard": build_dashboard,
    }


def _run_status() -> int:
    # Import manifest validation and stage modules
    from ueba import config
    from ueba.pipeline import manifest

    modules = _stage_modules()
    print(f"Pipeline status — model version {config.MODEL_VERSION}  (root: {config.BASE_DIR})\n")

    # Build a list of file paths each stage should produce
    checks: list[tuple[str, list[str]]] = [
        ("preprocess", modules["preprocess"].produces()),
        ("train-ae", modules["train-ae"].produces()),
        ("train-if", modules["train-if"].produces()),
        ("explain --split train", modules["explain"].produces("train")),
        ("explain --split test_stream", modules["explain"].produces("test_stream")),
        ("calibrate", modules["calibrate"].produces()),
        ("build-alerts --split main", modules["build-alerts"].produces("main")),
        ("build-alerts --split calib", modules["build-alerts"].produces("calib")),
        ("build-alerts --split test", modules["build-alerts"].produces("test")),
        ("build-dashboard", modules["build-dashboard"].produces()),
    ]

    missing_total = 0
    for stage, paths in checks:
        print(f"  {stage}")
        for path in paths:
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"    [ok]      {os.path.relpath(path, config.BASE_DIR)}  ({size / 1e6:,.1f} MB)")
            else:
                missing_total += 1
                print(f"    [MISSING] {os.path.relpath(path, config.BASE_DIR)}")
        print()

    problems = manifest.validate_recorded()
    if problems:
        print("Recorded-manifest problems:")
        for p in problems:
            print(f"  [STALE]   {p}")
        print()

    if missing_total or problems:
        print(f"{missing_total} artifact(s) missing, {len(problems)} manifest problem(s).")
        return 1
    print("All expected artifacts present.")
    return 0


def _run_all(args) -> int:
    modules = _stage_modules()
    plan = [
        ("preprocess", {}),
        ("train-ae", {"epochs": 100}),
        ("train-if", {}),
        ("explain", {"split": "train"}),
        ("calibrate", {"thresholds_only": False}),
        ("build-alerts", {"split": "main", "allow_missing": False}),
        ("explain", {"split": "test_stream"}),
        ("build-alerts", {"split": "test", "allow_missing": False}),
        ("build-alerts", {"split": "calib", "allow_missing": False}),
        ("build-dashboard", {}),
    ]
    for stage, extra in plan:
        print(f"\n===== {stage} {extra or ''} =====")
        stage_args = argparse.Namespace(**extra)
        modules[stage].run(stage_args)
    return 0


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.version:
        os.environ["UEBA_MODEL_VERSION"] = str(args.version)

    from ueba.pipeline.manifest import MissingArtifactError

    try:
        if args.stage == "status":
            return _run_status()
        if args.stage == "all":
            return _run_all(args)
        _stage_modules()[args.stage].run(args)
        return 0
    except MissingArtifactError as exc:
        print(f"\n[pipeline] FAILED — {args.stage}\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
