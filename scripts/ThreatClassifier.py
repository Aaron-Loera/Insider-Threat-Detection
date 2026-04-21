"""
ThreatClassifier.py — Rule-based threat scenario classifier.

Maps an alert's top contributing features to a predefined threat category,
transforming generic anomaly scores into analyst-friendly threat scenarios.
"""

# ---------------------------------------------------------------------------
# Threat category → trigger feature sets
# Extend this dict to add new scenarios without touching classify_threat().
# ---------------------------------------------------------------------------
THREAT_FEATURE_MAP: dict[str, set[str]] = {
    "Exfiltration: Removable Media": {
        "usb_insert_count",
        "usb_remove_count",
        "file_copy_count",
        "off_hours_usb_usage",
        "usb_file_activity_flag",
    },
    "Exfiltration: Email": {
        "external_emails_sent",
        "attachments_sent",
        "unique_recipients",
        "off_hours_emails",
        "external_comm_activity_flag",
    },
    "Exfiltration: Cloud/Web": {
        "http_upload_count",
        "http_cloud_storage_visits",
        "cloud_upload_flag",
        "suspicious_upload_flag",
    },
    "Unauthorized Access": {
        "off_hours_logon",
        "non_primary_pc_used_flag",
        "non_primary_pc_risk_flag",
        "pcs_used_count",
    },
    "Suspicious Browsing": {
        "http_jobsite_visits",
        "http_suspicious_site_visits",
        "http_long_url_count",
        "jobsite_usb_activity_flag",
    },
}

# Returned when no single category has a clear lead
FALLBACK: str = "General Anomaly"


def _normalize(feature: str) -> str:
    """
    Normalises a feature name for robust matching.
    Strips _zscore and _rolling_delta suffixes added during preprocessing,
    then lowercases, so the classifier is resilient to enriched column names.
    """
    name = feature.lower().strip()
    for suffix in ("_zscore", "_rolling_delta"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def classify_threat(top_contributors: list) -> str:
    """
    Classifies an alert's threat scenario from its top contributing features.

    For each threat category, counts how many of the incoming features appear
    in that category's trigger set. The category with the highest unique overlap
    count wins. Returns FALLBACK ("General Anomaly") when overlap is zero or
    when two or more categories share the top score (no single scenario dominates).

    Args:
        top_contributors: List of (feature_name, contribution_value) tuples,
                          as produced by AlertObjectBuilder.extract_top_contributors().

    Returns:
        str: The matched threat category, or "General Anomaly" if unclear.
    """
    if not top_contributors:
        return FALLBACK

    # Normalise incoming feature names
    incoming = {_normalize(feat) for feat, _ in top_contributors}

    # Score each category by feature overlap
    scores = {
        category: len(incoming & feature_set)
        for category, feature_set in THREAT_FEATURE_MAP.items()
    }

    max_overlap = max(scores.values())

    # No features matched any known category
    if max_overlap == 0:
        return FALLBACK

    # Collect all categories tied at the top overlap count
    winners = [cat for cat, count in scores.items() if count == max_overlap]

    # A tie means no single scenario dominates — fall back
    if len(winners) > 1:
        return FALLBACK

    return winners[0]