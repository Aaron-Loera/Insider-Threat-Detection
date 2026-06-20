"""Feature-name humanization and alert-summary text — pure, streamlit-free.

Maps raw model feature names (e.g. ``http_upload_count_zscore``) to analyst-readable
phrases and turns an alert's top-contributors blob into a one-sentence summary.
Extracted verbatim from dashboard/app.py so it can be unit-tested in isolation.
"""

import ast

_FEATURE_LABELS: dict[str, str] = {
    # ── Authentication ──
    "off_hours_logon":                          "after-hours logon activity",
    "off_hours_logon_count":                    "after-hours logon activity",
    "logon_count":                              "elevated logon frequency",
    # ── File activity ──
    "file_open_count":                          "high file open activity",
    "file_write_count":                         "file write activity",
    "file_copy_count":                          "file copy activity",
    "file_delete_count":                        "file deletion activity",
    "unique_files_accessed":                    "access to many unique files",
    "off_hours_files_accessed":                 "after-hours file access",
    # ── USB ──
    "usb_insert_count":                         "USB device insertion",
    "usb_remove_count":                         "USB device removals",
    "off_hours_usb_usage":                      "after-hours USB usage",
    "usb_file_activity_flag":                   "USB-related file activity",
    "jobsite_usb_activity_flag":                "job-site browsing combined with USB activity",
    # ── Email ──
    "emails_sent":                              "high email volume",
    "unique_recipients":                        "many unique email recipients",
    "external_emails":                          "external email activity",
    "external_email_count":                     "external email activity",
    "attachements_sent":                        "email attachment activity",
    "attachments_sent":                         "email attachment activity",
    "off_hours_emails":                         "after-hours email activity",
    # ── Cross-channel flags ──
    "off_hours_activity_flag":                  "off-hours behavioral anomalies",
    "external_comm_activity_flag":              "external communication anomalies",
    # ── HTTP / Web activity (raw counts) ──
    "http_total_requests":                      "total web requests",
    "http_visit_count":                         "web page visits",
    "http_download_count":                      "web downloads",
    "http_upload_count":                        "web uploads",
    "http_jobsite_visits":                      "job-site website visits",
    "http_cloud_storage_visits":                "cloud storage website visits",
    "http_suspicious_site_visits":              "suspicious website visits",
    "off_hours_http_requests":                  "after-hours HTTP activity",
    "http_long_url_count":                      "long-URL HTTP activity",
    "unique_domains_visited":                   "unique websites visited",
    # ── HTTP cross-channel flags ──
    "suspicious_upload_flag":                   "suspicious web upload activity",
    "cloud_upload_flag":                        "cloud storage upload activity",
    # ── Known z-score variants ──
    "file_delete_count_zscore":                 "unusually high file deletion activity",
    "file_write_count_zscore":                  "unusually high file write activity",
    "file_copy_count_zscore":                   "unusually high file copy activity",
    "file_open_count_zscore":                   "unusually high file open activity",
    "unique_files_accessed_zscore":             "unusually high unique file access",
    "off_hours_files_accessed_zscore":          "unusually high after-hours file access",
    "off_hours_logon_zscore":                   "unusually high after-hours logon activity",
    "http_long_url_count_zscore":               "unusually high long-URL HTTP activity",
    "off_hours_http_requests_zscore":           "unusually high after-hours HTTP activity",
    "attachments_sent_zscore":                  "unusually high attachment-sending activity",
    "attachements_sent_zscore":                 "unusually high attachment-sending activity",
    "external_emails_sent_zscore":              "unusually high external email activity",
    "external_email_count_zscore":              "unusually high external email activity",
    "emails_sent_zscore":                       "unusually high email volume",
    "unique_recipients_zscore":                 "unusually high number of unique email recipients",
    "usb_insert_count_zscore":                  "unusually high USB insertion activity",
    "http_total_requests_zscore":               "unusually high web request volume",
    "http_visit_count_zscore":                  "unusually high web browsing activity",
    "http_download_count_zscore":               "unusually high web download activity",
    "http_upload_count_zscore":                 "unusually high web upload activity",
    "http_jobsite_visits_zscore":               "unusually high job-site browsing activity",
    "http_cloud_storage_visits_zscore":         "unusually high cloud storage activity",
    "http_suspicious_site_visits_zscore":       "unusually high suspicious site visits",
    "unique_domains_visited_zscore":            "unusually high number of unique websites visited",
    # ── Known rolling-delta variants ──
    "file_delete_count_rolling_delta":          "sudden spike in file deletions",
    "file_write_count_rolling_delta":           "sudden spike in file write activity",
    "file_copy_count_rolling_delta":            "sudden spike in file copy activity",
    "file_open_count_rolling_delta":            "sudden spike in file open activity",
    "unique_files_accessed_rolling_delta":      "sudden increase in unique file access",
    "off_hours_files_accessed_rolling_delta":   "sudden increase in after-hours file access",
    "off_hours_logon_rolling_delta":            "sudden increase in after-hours logons",
    "emails_sent_rolling_delta":                "sudden spike in email volume",
    "external_emails_rolling_delta":            "sudden spike in external email activity",
    "attachments_sent_rolling_delta":           "sudden spike in attachment-sending",
    "attachements_sent_rolling_delta":          "sudden spike in attachment-sending",
    "usb_insert_count_rolling_delta":           "sudden spike in USB device activity",
    "http_requests_rolling_delta":              "sudden spike in HTTP requests",
    "off_hours_http_requests_rolling_delta":    "sudden increase in after-hours HTTP activity",
    "http_total_requests_rolling_delta":        "sudden spike in web requests",
    "http_visit_count_rolling_delta":           "sudden spike in web browsing",
    "http_download_count_rolling_delta":        "sudden spike in web downloads",
    "http_upload_count_rolling_delta":          "sudden spike in web uploads",
    "http_jobsite_visits_rolling_delta":        "sudden increase in job-site browsing",
    "http_cloud_storage_visits_rolling_delta":  "sudden increase in cloud storage website visits",
    "http_suspicious_site_visits_rolling_delta": "sudden increase in suspicious site visits",
    "http_long_url_count_rolling_delta":        "sudden spike in long-URL HTTP activity",
    "unique_domains_visited_rolling_delta":     "sudden increase in unique websites visited",
}

# Base-name phrases used by the pattern-matching fallback in prettify_feature_name()
_BASE_LABELS: dict[str, str] = {
    "file_delete_count":        "file deletions",
    "file_write_count":         "file write activity",
    "file_copy_count":          "file copy activity",
    "file_open_count":          "file open activity",
    "unique_files_accessed":    "unique file access",
    "off_hours_files_accessed": "after-hours file access",
    "off_hours_logon":          "after-hours logons",

    "logon_count":              "logon frequency",
    "logoff_count":             "logoff activity",
    "external_emails":          "external email activity",
    "external_email_count":     "external email activity",
    "external_emails_sent":     "external email activity",
    "http_long_url":            "long-URL HTTP activity",
    "off_hours_http":           "after-hours HTTP activity",
    "attachments_sent":         "attachment-sending activity",
    "attachements_sent":        "attachment-sending",
    "emails_sent":              "email volume",
    "unique_recipients":        "unique email recipients",
    "off_hours_emails":         "after-hours email activity",
    "usb_insert_count":         "USB device activity",
    "usb_remove_count":         "USB device removals",
    "off_hours_usb_usage":      "after-hours USB usage",
    "http_requests":            "HTTP requests",
    "http_total_requests":          "web requests",
    "http_visit_count":             "web page visits",
    "http_download_count":          "web downloads",
    "http_upload_count":            "web uploads",
    "http_jobsite_visits":          "job-site browsing",
    "http_cloud_storage_visits":    "cloud storage website visits",
    "http_suspicious_site_visits":  "suspicious site visits",
    "unique_domains_visited":       "unique websites visited",
}


def _humanize_base(base: str) -> str:
    """Map a bare feature base name to a readable phrase for pattern-generated sentences."""
    return _BASE_LABELS.get(base, base.replace("_", " "))


def parse_top_contributors(raw) -> list[str]:
    """Return a list of feature name strings from top_contributors, however it is stored."""
    if raw is None:
        return []
    if isinstance(raw, float):
        return []  # NaN from a left-join miss
    if isinstance(raw, list):
        names = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                names.append(str(item[0]))
            elif isinstance(item, str):
                names.append(item)
        return names
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parse_top_contributors(parsed)
        except Exception:
            pass
    return []


def parse_top_contributors_with_values(raw) -> list[tuple]:
    """Return list of (feature_name, contribution_value) tuples from top_contributors."""
    if raw is None:
        return []
    if isinstance(raw, float):
        return []  # NaN from a left-join miss
    if isinstance(raw, list):
        pairs = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), item[1]))
            elif isinstance(item, (list, tuple)) and len(item) == 1:
                pairs.append((str(item[0]), None))
            elif isinstance(item, str):
                pairs.append((item, None))
        return pairs
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parse_top_contributors_with_values(parsed)
        except Exception:
            pass
    return []


def prettify_feature_name(name: str) -> str:
    """Convert a raw feature name to an analyst-readable investigation phrase."""
    cleaned = name.strip().replace("contribution_", "")
    # 1. Exact dictionary hit — covers flags, known features, and common variants
    if cleaned in _FEATURE_LABELS:
        return _FEATURE_LABELS[cleaned]
    # 2. Pattern: z-score suffix → "unusually high <base>"
    if cleaned.endswith("_zscore"):
        base = cleaned[: -len("_zscore")]
        return f"unusually high {_humanize_base(base)}"
    # 3. Pattern: rolling-delta suffix → "sudden spike/increase in <base>"
    if cleaned.endswith("_rolling_delta"):
        base = cleaned[: -len("_rolling_delta")]
        human = _humanize_base(base)
        if base.endswith(("_count", "_sent")):
            return f"sudden spike in {human}"
        return f"sudden increase in {human}"
    # 4. Last resort: replace underscores with spaces
    return cleaned.replace("_", " ")


def build_alert_summary(top_contributors_raw) -> str:
    """
    Return a short, analyst-friendly sentence describing the top contributing
    behaviors for an alert. Shows up to 3 contributors; appends 'and additional
    signals' when the full list contains more than 3 unique contributors.
    """
    features = parse_top_contributors(top_contributors_raw)
    if not features:
        return "No contributor detail available for this alert."

    labels: list[str] = []
    seen: set[str] = set()
    for f in features:
        lbl = prettify_feature_name(f)
        if lbl not in seen:
            seen.add(lbl)
            labels.append(lbl)

    has_more = len(labels) > 3
    display = labels[:3]

    if len(display) == 1:
        return f"This alert is primarily driven by {display[0]}."

    if len(display) == 2:
        return f"This alert is mainly driven by {display[0]} and {display[1]}."

    # Exactly 3 shown
    body = f"{display[0]}, {display[1]}, and {display[2]}"
    suffix = ", and additional signals" if has_more else ""
    return f"This alert is mainly driven by {body}{suffix}."
