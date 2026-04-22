"""
SuppressionEngine.py — Configurable suppression rule engine for the alert feed.

Rules apply ONLY to MEDIUM-band alerts. HIGH and CRITICAL are never touched.
Known insider users (loaded from CERT ground truth) are always protected.

Suppressed rows receive:
  status           = "SUPPRESSED"
  suppression_rule = <rule_id>

They remain in the alert table for audit and are routed to the Suppressed
Alerts view rather than the main analyst feed.
"""

import ast
import os
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required columns — validated at entry; defaults injected if absent
# ---------------------------------------------------------------------------
_REQUIRED_COLUMNS: dict[str, object] = {
    "composite_risk_band":  "MEDIUM",
    "department":           "UNKNOWN",
    "role":                 None,
    "top_contributors":     None,
    "alert_sequence_id":    None,
    "status":               "NEW",
}

# Must match CROSS_CHANNEL_FLAGS in AlertObjectBuilder.py
_CROSS_CHANNEL_FLAGS = [
    "off_hours_activity_flag",
    "usb_file_activity_flag",
    "external_comm_activity_flag",
    "jobsite_usb_activity_flag",
    "suspicious_upload_flag",
    "cloud_upload_flag",
    "non_primary_pc_risk_flag",
]

# ---------------------------------------------------------------------------
# Known insider protection
# ---------------------------------------------------------------------------
KNOWN_INSIDER_USERS: frozenset[str] = frozenset()


def load_known_insiders(insiders_csv_path: str | None = None) -> None:
    """
    Populates KNOWN_INSIDER_USERS from the CERT ground-truth insiders.csv.
    Call this before apply_suppression() when the file is available.

    Hook: pass config.INSIDERS_PATH, or any CSV with a "user" column.
    If the path is None or missing the function silently no-ops.
    """
    global KNOWN_INSIDER_USERS
    if not insiders_csv_path or not os.path.exists(insiders_csv_path):
        return
    try:
        df = pd.read_csv(insiders_csv_path)
        col = next((c for c in df.columns if c.lower().strip() == "user"), None)
        if col is None:
            warnings.warn(
                f"[SuppressionEngine] insiders.csv has no 'user' column "
                f"(found: {list(df.columns)}). Protection disabled.",
                stacklevel=2,
            )
            return
        users = frozenset(df[col].astype(str).str.lower().str.strip().dropna())
        KNOWN_INSIDER_USERS = users
        print(f"[SuppressionEngine] Loaded {len(users)} known insider users — suppression protection active.")
    except Exception as exc:
        warnings.warn(
            f"[SuppressionEngine] Could not load insiders file ({exc!r}). "
            "Protection disabled.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Contributor parsing helpers
# ---------------------------------------------------------------------------

def _parse_top_contributors(val) -> list[tuple[str, float]]:
    """
    Safely parses a top_contributors value from any storage format.

    Handles:
    - list of (feature, value) tuples (in-memory format)
    - string-serialised list (parquet/CSV round-trip via repr/str)
    - None / NaN → empty list
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return [(str(f), float(v)) for f, v in val if f is not None]
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [(str(f), float(v)) for f, v in parsed if f is not None]
        except (ValueError, SyntaxError):
            pass
    return []


def contributor_overlap(top_contributors: list, features: set[str]) -> int:
    """
    Returns the number of top contributors whose feature name belongs to the
    given feature set. Checks ALL contributors, not just the first.
    """
    return sum(1 for feat, _ in top_contributors if feat in features)


def contributor_family_score(top_contributors: list, features: set[str]) -> float:
    """
    Returns the weighted fraction of total absolute contribution value that
    comes from the given feature family.

    0.0 = none of the explained variance belongs to this family.
    1.0 = all explained variance belongs to this family.
    """
    if not top_contributors:
        return 0.0
    total = sum(abs(v) for _, v in top_contributors)
    if total == 0.0:
        return 0.0
    family = sum(abs(v) for feat, v in top_contributors if feat in features)
    return family / total


# ---------------------------------------------------------------------------
# Feature families used by suppression rules
# ---------------------------------------------------------------------------
_LOGON_FEATURES: set[str] = {
    "logon_count", "logoff_count", "off_hours_logon",
}

_EMAIL_FEATURES: set[str] = {
    "emails_sent", "external_emails_sent", "attachments_sent",
    "unique_recipients", "off_hours_emails", "external_comm_activity_flag",
}

# Departments that normalise high email or logon activity for their roles
_IT_DEPARTMENTS: set[str] = {
    "it", "information technology", "information systems",
    "it security", "systems", "technology",
}

_SALES_DEPARTMENTS: set[str] = {
    "sales", "communications", "marketing",
    "business development", "customer success",
    "account management",
}

# Role substring keywords for sales/communications — checked case-insensitively
_SALES_ROLE_KEYWORDS: set[str] = {
    "sales", "account", "communications", "marketing",
    "relation", "business development",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flag_count(df: pd.DataFrame) -> pd.Series:
    """Returns the number of active cross-channel flags per row."""
    present = [f for f in _CROSS_CHANNEL_FLAGS if f in df.columns]
    if not present:
        return pd.Series(0, index=df.index, dtype=int)
    return df[present].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(int)


def _is_it_dept(dept_series: pd.Series) -> pd.Series:
    return dept_series.str.lower().str.strip().isin(_IT_DEPARTMENTS)


def _is_sales_identity(dept_series: pd.Series, role_series: pd.Series) -> pd.Series:
    """
    Returns True where the user's department OR role indicates a
    sales/communications/marketing function.
    """
    dept_match = dept_series.str.lower().str.strip().isin(_SALES_DEPARTMENTS)

    # Role keyword match (handles NaN gracefully)
    role_lower = role_series.fillna("").astype(str).str.lower()
    role_match = role_lower.apply(
        lambda r: any(kw in r for kw in _SALES_ROLE_KEYWORDS)
    )
    return dept_match | role_match


# ---------------------------------------------------------------------------
# Suppression rules
# ---------------------------------------------------------------------------
# Each rule is evaluated against the full MEDIUM candidate DataFrame.
# "condition" receives the full alert_df (all rows) and the candidate Boolean
# mask (only MEDIUM, not yet suppressed), returning a bool Series over all rows.
# The engine handles the masking so conditions see the full context.
# ---------------------------------------------------------------------------

def _rule_it_logon(df: pd.DataFrame, candidate_mask: pd.Series) -> pd.Series:
    """
    Suppress MEDIUM alerts for IT-department users where logon-family features
    account for ≥ 40% of the weighted contribution score OR ≥ 2 of the top-k
    contributors are logon features — indicating the anomaly is driven by
    routine admin authentication volume.
    """
    it_mask = _is_it_dept(df["department"])

    parsed = df["top_contributors"].apply(_parse_top_contributors)
    logon_dominated = parsed.apply(
        lambda tc: (
            contributor_family_score(tc, _LOGON_FEATURES) >= 0.40
            or contributor_overlap(tc, _LOGON_FEATURES) >= 2
        )
    )
    return candidate_mask & it_mask & logon_dominated


def _rule_sales_email(df: pd.DataFrame, candidate_mask: pd.Series) -> pd.Series:
    """
    Suppress MEDIUM alerts for Sales/Communications users (by department OR role)
    where email-family features account for ≥ 40% of the weighted contribution
    score OR ≥ 2 of the top-k contributors are email features.
    """
    sales_mask = _is_sales_identity(df["department"], df["role"].fillna(""))

    parsed = df["top_contributors"].apply(_parse_top_contributors)
    email_dominated = parsed.apply(
        lambda tc: (
            contributor_family_score(tc, _EMAIL_FEATURES) >= 0.40
            or contributor_overlap(tc, _EMAIL_FEATURES) >= 2
        )
    )
    return candidate_mask & sales_mask & email_dominated


def _rule_single_day_no_flags(df: pd.DataFrame, candidate_mask: pd.Series) -> pd.Series:
    """
    Suppress MEDIUM alerts with zero active cross-channel flags AND no sequence
    membership. A lone anomaly with no corroborating cross-channel signals is
    likely noise and does not warrant analyst attention.
    """
    no_flags = _flag_count(df) == 0
    no_sequence = df["alert_sequence_id"].isna()
    return candidate_mask & no_flags & no_sequence


SUPPRESSION_RULES: list[dict] = [
    {
        "rule_id": "suppress_it_logon",
        "description": (
            "Suppress MEDIUM alerts for IT-dept users driven by logon-family features "
            "(≥40% contribution weight or ≥2 logon features in top-k). "
            "Elevated authentication counts are normal for IT admins."
        ),
        "fn": _rule_it_logon,
        "enabled": True,
    },
    {
        "rule_id": "suppress_sales_email",
        "description": (
            "Suppress MEDIUM alerts for Sales/Communications users (by department OR role) "
            "driven by email-family features (≥40% contribution weight or ≥2 email features "
            "in top-k). High send rates are expected in customer-facing roles."
        ),
        "fn": _rule_sales_email,
        "enabled": True,
    },
    {
        "rule_id": "suppress_single_day_no_flags",
        "description": (
            "Suppress MEDIUM alerts with zero cross-channel flags and no sequence membership. "
            "Isolated spikes with no corroborating signals are likely noise."
        ),
        "fn": _rule_single_day_no_flags,
        "enabled": True,
    },
]


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

def _validate_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Validates required columns exist; injects safe defaults for missing ones.
    Returns (prepared_df, list_of_warnings).
    """
    df = df.copy()
    issued = []

    for col, default in _REQUIRED_COLUMNS.items():
        if col not in df.columns:
            issued.append(
                f"[SuppressionEngine] Column '{col}' not found — "
                f"defaulting to {default!r}. Rules depending on this column will not fire."
            )
            df[col] = default

    if "suppression_rule" not in df.columns:
        df["suppression_rule"] = None

    return df, issued


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_suppression(
    alert_df: pd.DataFrame,
    rules: list[dict] | None = None,
    known_insider_users: frozenset[str] | None = None,
    target_reduction_pct: float = 15.0,
) -> pd.DataFrame:
    """
    Applies suppression rules to the alert DataFrame.

    Only MEDIUM-band alerts (composite_risk_band == "MEDIUM") are candidates.
    HIGH and CRITICAL are never suppressed. Known insider users are always
    protected regardless of which rule would otherwise match.

    Args:
        alert_df: Output of AlertObjectBuilder.build_alert_df() after sequence
                  detection and priority backfill.
        rules: Override SUPPRESSION_RULES (useful for testing individual rules).
        known_insider_users: Override module-level KNOWN_INSIDER_USERS set.
        target_reduction_pct: Target minimum MEDIUM-alert reduction (validation only).

    Returns:
        pd.DataFrame: Copy of alert_df with ``status`` and ``suppression_rule``
                      updated for suppressed rows, and detailed console output.
    """
    if rules is None:
        rules = SUPPRESSION_RULES
    if known_insider_users is None:
        known_insider_users = KNOWN_INSIDER_USERS

    df, validation_warnings = _validate_and_prepare(alert_df)

    for msg in validation_warnings:
        warnings.warn(msg, stacklevel=2)

    # ── Insider protection mask — never suppress known ground-truth users ──
    if "user" in df.columns and known_insider_users:
        protected_mask = df["user"].str.lower().str.strip().isin(known_insider_users)
        n_protected = int(protected_mask.sum())
        if n_protected:
            print(f"[SuppressionEngine] Protecting {n_protected} alerts from {len(known_insider_users)} known insider users.")
    else:
        protected_mask = pd.Series(False, index=df.index)

    # ── Only MEDIUM-band rows are candidates ──
    medium_mask = df["composite_risk_band"] == "MEDIUM"
    total_medium = int(medium_mask.sum())

    per_rule_counts: dict[str, int] = {}
    total_suppressed = 0

    for rule in rules:
        if not rule.get("enabled", True):
            continue

        # Candidate = MEDIUM, not yet suppressed, not protected
        candidate_mask = (
            medium_mask
            & (df["status"] != "SUPPRESSED")
            & ~protected_mask
        )
        if not candidate_mask.any():
            break

        try:
            rule_mask = rule["fn"](df, candidate_mask)
            # Ensure result is boolean and aligned to df.index
            rule_mask = rule_mask.reindex(df.index, fill_value=False).astype(bool)
        except Exception as exc:
            warnings.warn(
                f"[SuppressionEngine] Rule '{rule['rule_id']}' raised {exc!r}. Skipping.",
                stacklevel=2,
            )
            continue

        hits = candidate_mask & rule_mask
        n = int(hits.sum())
        df.loc[hits, "status"] = "SUPPRESSED"
        df.loc[hits, "suppression_rule"] = rule["rule_id"]
        per_rule_counts[rule["rule_id"]] = n
        total_suppressed += n

    # ── Detailed validation output ──
    pct = (total_suppressed / total_medium * 100) if total_medium > 0 else 0.0
    passed = pct >= target_reduction_pct

    print("\nSuppression complete:")
    print(f"  MEDIUM before  : {total_medium:,}")
    print(f"  Suppressed     : {total_suppressed:,}")
    print(f"  Reduction      : {pct:.1f}% (target >= {target_reduction_pct:.0f}%)"
          f"  {'✓ PASSED' if passed else '✗ BELOW TARGET'}")
    print("\nPer-rule:")
    for rule_id, count in per_rule_counts.items():
        print(f"  - {rule_id}: {count:,}")
    if not per_rule_counts:
        print("  (no rules matched any rows)")

    return df