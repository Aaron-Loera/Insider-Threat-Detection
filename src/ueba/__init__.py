"""UEBA insider-threat detection package.

Layout:
    ueba.config     — central path/version configuration (paths.local.py overrides)
    ueba.constants  — feature-engineering constants (work hours, domain lists, dtypes)
    ueba.risk       — single source of truth for risk bands and percentile ranking
    ueba.features   — CERT log preprocessing / Layer A+B feature engineering
    ueba.models     — autoencoder, isolation forest, training-data preparation
    ueba.alerts     — alert object building and reconstruction-error explanation
    ueba.serving    — live scoring simulation, dashboard dataset build/upload
    ueba.viz        — analysis/visualization helpers used by the notebooks
"""

__version__ = "0.1.0"
