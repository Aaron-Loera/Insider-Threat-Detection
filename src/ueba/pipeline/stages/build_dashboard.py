"""Stage: build-dashboard — slim serving parquet for Streamlit Cloud.

Wraps ueba.serving.build_dashboard_dataset (the v6 slim serving layer) with
the pipeline's manifest contract.
"""

from ueba import config
from ueba.pipeline import manifest

STAGE = "build-dashboard"


def requires() -> list[tuple[str, str]]:
    return [
        (config.UEBA_B_PATH, "preprocess"),
        (config.ANALYST_TABLE_PARQUET, "build-alerts --split main"),
    ]


def produces() -> list[str]:
    return [config.DASHBOARD_PARQUET]


def run(args) -> None:
    import sys

    from ueba.serving import build_dashboard_dataset

    manifest.require(requires())
    # The serving CLI parses sys.argv; run it with a clean argv.
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        build_dashboard_dataset.main()
    finally:
        sys.argv = argv
    manifest.record(STAGE, produces())
