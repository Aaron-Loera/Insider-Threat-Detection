"""Unit tests for the dashboard label/summary helpers (dashboard/lib/labels.py).

These are pure, streamlit-free functions, so they import directly from the lib
module (via the dashboard namespace package) without booting the Streamlit app.
"""

import pytest

from dashboard.lib import labels


# ── parse_top_contributors ──────────────────────────────────────────────────
class TestParseTopContributors:
    def test_list_of_pairs(self):
        raw = [("logon_count", 1.0), ("file_copy_count", 0.5)]
        assert labels.parse_top_contributors(raw) == ["logon_count", "file_copy_count"]

    def test_list_of_strings(self):
        assert labels.parse_top_contributors(["a", "b"]) == ["a", "b"]

    def test_stringified_list(self):
        raw = "[('logon_count', 1.0), ('emails_sent', 0.3)]"
        assert labels.parse_top_contributors(raw) == ["logon_count", "emails_sent"]

    def test_none_returns_empty(self):
        assert labels.parse_top_contributors(None) == []

    def test_nan_returns_empty(self):
        assert labels.parse_top_contributors(float("nan")) == []

    def test_empty_string_returns_empty(self):
        assert labels.parse_top_contributors("") == []
        assert labels.parse_top_contributors("   ") == []

    def test_unparseable_string_returns_empty(self):
        assert labels.parse_top_contributors("not a python literal") == []

    def test_non_list_literal_returns_empty(self):
        # A valid literal that isn't a list (e.g. a dict) falls through to [].
        assert labels.parse_top_contributors("{'a': 1}") == []


# ── parse_top_contributors_with_values ──────────────────────────────────────
class TestParseTopContributorsWithValues:
    def test_pairs_keep_values(self):
        raw = [("logon_count", 1.0), ("file_copy_count", 0.5)]
        assert labels.parse_top_contributors_with_values(raw) == [
            ("logon_count", 1.0),
            ("file_copy_count", 0.5),
        ]

    def test_single_element_item_gets_none_value(self):
        assert labels.parse_top_contributors_with_values([("logon_count",)]) == [
            ("logon_count", None)
        ]

    def test_bare_string_item_gets_none_value(self):
        assert labels.parse_top_contributors_with_values(["logon_count"]) == [
            ("logon_count", None)
        ]

    def test_stringified(self):
        raw = "[('http_upload_count', 0.9)]"
        assert labels.parse_top_contributors_with_values(raw) == [("http_upload_count", 0.9)]

    @pytest.mark.parametrize("bad", [None, float("nan"), "", "garbage"])
    def test_bad_inputs_return_empty(self, bad):
        assert labels.parse_top_contributors_with_values(bad) == []


# ── prettify_feature_name ───────────────────────────────────────────────────
class TestPrettifyFeatureName:
    def test_exact_dictionary_hit(self):
        assert labels.prettify_feature_name("logon_count") == "elevated logon frequency"

    def test_contribution_prefix_stripped(self):
        assert labels.prettify_feature_name("contribution_logon_count") == "elevated logon frequency"

    def test_zscore_known_variant_uses_dictionary(self):
        # _FEATURE_LABELS has an explicit entry that wins over the pattern.
        assert labels.prettify_feature_name("http_upload_count_zscore") == (
            "unusually high web upload activity"
        )

    def test_zscore_pattern_fallback(self):
        # A base not in _FEATURE_LABELS but in _BASE_LABELS → pattern phrasing.
        assert labels.prettify_feature_name("http_requests_zscore") == (
            "unusually high HTTP requests"
        )

    def test_rolling_delta_count_base_is_spike(self):
        assert labels.prettify_feature_name("http_requests_rolling_delta") == (
            "sudden spike in HTTP requests"
        )

    def test_rolling_delta_non_count_base_is_increase(self):
        # off_hours_logon does not end in _count/_sent → "sudden increase".
        assert labels.prettify_feature_name("off_hours_logon_rolling_delta") == (
            "sudden increase in after-hours logons"
        )

    def test_unknown_feature_underscores_to_spaces(self):
        assert labels.prettify_feature_name("totally_unknown_feature") == "totally unknown feature"


# ── build_alert_summary ─────────────────────────────────────────────────────
class TestBuildAlertSummary:
    def test_no_contributors(self):
        assert labels.build_alert_summary(None) == (
            "No contributor detail available for this alert."
        )

    def test_one_contributor(self):
        out = labels.build_alert_summary([("logon_count", 1.0)])
        assert out == "This alert is primarily driven by elevated logon frequency."

    def test_two_contributors(self):
        out = labels.build_alert_summary([("logon_count", 1), ("file_copy_count", 1)])
        assert out == (
            "This alert is mainly driven by elevated logon frequency and file copy activity."
        )

    def test_three_contributors_no_suffix(self):
        out = labels.build_alert_summary(
            [("logon_count", 1), ("file_copy_count", 1), ("emails_sent", 1)]
        )
        assert out == (
            "This alert is mainly driven by elevated logon frequency, file copy activity, "
            "and high email volume."
        )
        assert "additional signals" not in out

    def test_four_plus_contributors_gets_suffix(self):
        out = labels.build_alert_summary(
            [
                ("logon_count", 1),
                ("file_copy_count", 1),
                ("emails_sent", 1),
                ("usb_insert_count", 1),
            ]
        )
        assert out.endswith(", and additional signals.")

    def test_duplicate_labels_deduped(self):
        # external_emails and external_email_count map to the same phrase, so two
        # raw contributors collapse to one displayed label (singular phrasing).
        out = labels.build_alert_summary([("external_emails", 1), ("external_email_count", 1)])
        assert out == "This alert is primarily driven by external email activity."
