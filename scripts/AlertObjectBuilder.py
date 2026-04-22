import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scripts.ThreatClassifier import classify_threat

# Cross-channel behavioral flags defined in Preprocessing.apply_ueba_enhancements
CROSS_CHANNEL_FLAGS = [
    "off_hours_activity_flag",
    "usb_file_activity_flag",
    "external_comm_activity_flag",
    "jobsite_usb_activity_flag",
    "suspicious_upload_flag",
    "cloud_upload_flag",
    "non_primary_pc_risk_flag",
]
TOTAL_FLAGS = len(CROSS_CHANNEL_FLAGS)

# Role sensitivity → normalized 0-1 contribution
ROLE_SENSITIVITY_MAP = {"CRITICAL": 1.0, "HIGH": 0.75, "MEDIUM": 0.5, "LOW": 0.25}

class AlertObjectBuilder:
    """
    Converts raw reconstruction error outputs into contextual, SOC-ready alert objects.
    Operates on both Autoencoder reconstruction errors and Isolation Forest anomaly scores,
    producing a unified alert with per-signal percentile ranks and risk bands.
    """

    def __init__(self, percentile_thresholds: dict | None=None, top_k: int=3) -> None:
        """
        Initializing the alert object builder.

        Args:
            percentile_thresholds: The risk band thresholds (shared by both AE and IF signals)
            top_k: Number of top contributing features to extract

        Returns:
            None:
        """
        if percentile_thresholds is None:
            percentile_thresholds = {
                "LOW": 80,
                "MEDIUM": 90,
                "HIGH": 95,
                "CRITICAL": 100
            }

        self.percentile_thresholds = percentile_thresholds
        self.top_k = top_k
        self.ae_baseline = None
        self.if_baseline = None


    def fit_ae_baseline(self, reconstruction_errors: np.ndarray) -> None:
        """
        Stores the baseline Autoencoder reconstruction error distribution.

        Args:
            reconstruction_errors: Array of baseline reconstruction errors.

        Returns:
            None:
        """
        self.ae_baseline = np.sort(reconstruction_errors)


    def fit_if_baseline(self, anomaly_scores: np.ndarray) -> None:
        """
        Stores the baseline Isolation Forest anomaly score distribution.

        Args:
            anomaly_scores: Array of baseline Isolation Forest anomaly scores.

        Returns:
            None:
        """
        self.if_baseline = np.sort(anomaly_scores)


    def compute_ae_percentile(self, error: float) -> float:
        """
        Computes the percentile rank of an Autoencoder reconstruction error value.

        Args:
            error: The reconstruction error value to rank

        Returns:
            float: The percentile the error value falls within (0-100)
        """
        if self.ae_baseline is None:
            raise ValueError("AE baseline distribution not fitted. Call fit_ae_baseline() first.")

        percentile = np.searchsorted(self.ae_baseline, error) / len(self.ae_baseline) * 100
        return percentile


    def compute_if_percentile(self, score: float) -> float:
        """
        Computes the percentile rank of an Isolation Forest anomaly score.

        Args:
            score: The anomaly score value to rank

        Returns:
            float: The percentile the score falls within (0-100)
        """
        if self.if_baseline is None:
            raise ValueError("IF baseline distribution not fitted. Call fit_if_baseline() first.")

        percentile = np.searchsorted(self.if_baseline, score) / len(self.if_baseline) * 100
        return percentile


    def assign_risk_band(self, percentile: float) -> str:
        """
        Assigns a risk band based on the provided percentile. Shared by both AE and IF signals.

        Args:
            percentile: The assigned percentile of an error or score value

        Returns:
            str: The risk band level
        """
        for label, thresh in self.percentile_thresholds.items():
            if percentile <= thresh:
                return label

        return "CRITICAL"


    def compute_priority(self, row: pd.Series) -> float:
        """
        Computes a composite priority score (0-100) that combines four signals:

        - max(ae_percentile_rank, if_percentile_rank) / 100  — weight 0.50
        - cross_channel_flag_count / total_flags             — weight 0.20
        - role_sensitivity (normalized 0-1)                  — weight 0.15
        - sequence_membership_bonus (1 if alert_sequence_id  — weight 0.15
          is set, 0 otherwise)

        Args:
            row: A pd.Series containing at minimum ae_percentile_rank and
                 if_percentile_rank. Enriched rows may also carry cross-channel
                 flag columns, role_sensitivity, and alert_sequence_id.

        Returns:
            float: Composite priority score in the range [0, 100].
        """
        # Component 1 — peak model signal (0-1)
        # Use pd.isna() rather than `or 0`: bool(nan) is True in Python so `nan or 0` returns nan.
        _ae = row.get("ae_percentile_rank", 0)
        _if = row.get("if_percentile_rank", 0)
        ae_pct = 0.0 if pd.isna(_ae) else float(_ae)
        if_pct = 0.0 if pd.isna(_if) else float(_if)
        signal_score = max(ae_pct, if_pct) / 100.0

        # Component 2 — cross-channel flag density (0-1)
        present = [f for f in CROSS_CHANNEL_FLAGS if f in row.index]
        if present:
            flag_score = sum(bool(row[f]) for f in present) / TOTAL_FLAGS
        else:
            flag_score = 0.0

        # Component 3 — role sensitivity (0-1)
        rs = row.get("role_sensitivity", None)
        if isinstance(rs, str):
            sensitivity_score = ROLE_SENSITIVITY_MAP.get(rs.upper(), 0.25)
        elif not pd.isna(rs):
            sensitivity_score = float(np.clip(float(rs), 0.0, 1.0))
        else:
            sensitivity_score = 0.25

        # Component 4 — sequence membership bonus (binary 0 or 1)
        # pd.isna covers None, float nan, and pd.NA — isinstance(pd.NA, float) is False so the
        # old guard missed it, treating pd.NA as a valid sequence ID.
        seq_id = row.get("alert_sequence_id", None)
        in_sequence = 0.0 if pd.isna(seq_id) else 1.0

        raw = 0.50 * signal_score + 0.20 * flag_score + 0.15 * sensitivity_score + 0.15 * in_sequence
        return round(raw * 100.0, 4)


    def extract_top_contributors(self, row: pd.Series) -> list:
        """
        Extracts the top-K contributing factors from a row.

        Args:
            row: A row consisting of metadata and reconstruction error data

        Returns:
            list: A list of the form: (feature_name, contribution_value)
        """
        # Extracting the contribution-related columns
        contribution_cols = [col for col in row.index if col.startswith("contribution_")]
        contributions = row[contribution_cols].sort_values(ascending=False)

        # Finding the top-K contributing features
        top_features = contributions.head(self.top_k)

        top_contributors = [(feature.replace("contribution_", ""), value) for feature, value in top_features.items()]
        return top_contributors


    def build_alert_from_row(self, row: pd.Series, w1: float=0.5, w2: float=0.5) -> dict:
        """
        Builds an alert dictionary for a single sample using both AE and IF signals.
        Includes z-score and rolling delta context in the explanation when available.

        Args:
            row: A row consisting of metadata, reconstruction error data, and IF anomaly score.
                 Optionally includes {feature}_zscore and {feature}_rolling_delta columns for
                 richer explanation text. Also optionally includes LDAP identity fields:
                 department, role, functional_unit, supervisor, role_sensitivity.
            w1: The weight to assign to AE percentile
            w2: The weight to assign to IF percentile
        Returns:
            dict: A structured alert object
        """
        if w1 + w2 > 1.0:
            raise ValueError(f"w1 and 2 must add up to 1.0. Sum is {w1+w2}")

        # Computing AE percentile and risk band
        ae_error = row["total_reconstruction_error"]
        ae_percentile = self.compute_ae_percentile(ae_error)
        ae_risk_band = self.assign_risk_band(ae_percentile)
        top_features = self.extract_top_contributors(row)
        threat_category = classify_threat(top_features)

        # Computing IF percentile and risk band
        if_score = row["if_anomaly_score"]
        if_percentile = self.compute_if_percentile(if_score)
        if_risk_band = self.assign_risk_band(if_percentile)

        # Computing composite signal (equal-weight fusion of AE and IF percentile ranks)
        composite_score = w1 * ae_percentile + w2 * if_percentile
        composite_risk_band = self.assign_risk_band(composite_score)
        both_signals_high = ae_risk_band in ("HIGH", "CRITICAL") and if_risk_band in ("HIGH", "CRITICAL")

        # Building per-feature narrative with z-score context when available
        narrative_parts = []
        for feat, contrib_val in top_features:
            zscore_col = f"{feat}_zscore"
            delta_col = f"{feat}_rolling_delta"
            if zscore_col in row.index and not pd.isna(row[zscore_col]):
                z = row[zscore_col]
                delta = row[delta_col] if delta_col in row.index and not pd.isna(row[delta_col]) else None
                delta_str = f", {delta:+.1f} vs 5-day avg" if delta is not None else ""
                narrative_parts.append(f"{feat} (z={z:.1f}{delta_str}, contrib={contrib_val:.2f})")
            else:
                narrative_parts.append(f"{feat} ({contrib_val:.2f})")

        explanation = (
            f"AE deviation at {ae_percentile:.1f}th pct ({ae_risk_band}); "
            f"IF anomaly at {if_percentile:.1f}th pct ({if_risk_band}). "
            f"Composite: {composite_risk_band} ({composite_score:.1f}th pct). "
            f"Drivers: " + "; ".join(narrative_parts)
        )

        # Extracting optional LDAP identity fields
        _ldap_fields = ("department", "role", "functional_unit", "supervisor", "role_sensitivity")
        ldap = {f: (row[f] if f in row.index and not pd.isna(row[f]) else None) for f in _ldap_fields}

        # Creating alert dictionary (alert_sequence_id populated later by aggregate_alerts)
        alert = {
            "alert_sequence_id": None,
            "user": row["user"],
            "day": row["day"],
            "department": ldap["department"] if ldap["department"] is not None else "UNKNOWN",
            "role": ldap["role"],
            "functional_unit": ldap["functional_unit"],
            "supervisor": ldap["supervisor"],
            "role_sensitivity": ldap["role_sensitivity"],
            "ae_percentile_rank": ae_percentile,
            "ae_risk_band": ae_risk_band,
            "top_contributors": top_features,
            "threat_category": threat_category,
            "if_anomaly_score": if_score,
            "if_percentile_rank": if_percentile,
            "if_risk_band": if_risk_band,
            "composite_score": composite_score,
            "composite_risk_band": composite_risk_band,
            "both_signals_high": both_signals_high,
            "status": "NEW",
            "explanation": explanation,
        }

        # Compute composite_priority from the fully-assembled alert so all fields are available
        alert["composite_priority"] = self.compute_priority(pd.Series(alert | {
            f: row[f] for f in CROSS_CHANNEL_FLAGS if f in row.index
        }))

        return alert


    def build_alert_df(self, explanation_df: pd.DataFrame, w1: float=0.5, w2: float=0.5) -> pd.DataFrame:
        """
        Generates an alert DataFrame from an aggregated table of reconstruction errors and anomaly
        scores. Uses vectorized numpy/pandas operations instead of row-by-row iteration.

        Args:
            explanation_df: The enriched DataFrame containing AE reconstruction errors, IF anomaly
                            scores, and optionally {feature}_zscore / {feature}_rolling_delta columns
                            for richer explanation text.
            w1: The weight to assign to AE percentile
            w2: The weight to assign to IF percentile

        Returns:
            pd.DataFrame: An alert-ready structured DataFrame
        """
        if w1 + w2 != 1.0:
            raise ValueError(f"w1 and w2 must add up to 1.0. Sum is {w1 + w2}")

        df = explanation_df.reset_index(drop=True)
        n = len(df)

        # Extracting optional LDAP identity columns when present in the input
        _ldap_fields = ("department", "role", "functional_unit", "supervisor", "role_sensitivity")
        ldap_cols = {f: df[f].values if f in df.columns else np.full(n, None, dtype=object) for f in _ldap_fields}

        # department must never be null; fall back to UNKNOWN for unmatched users
        dept = ldap_cols["department"].copy()
        null_mask = pd.isnull(dept)
        dept[null_mask] = "UNKNOWN"
        ldap_cols["department"] = dept

        # Vectorized percentile computation
        ae_errors = df["total_reconstruction_error"].values
        if_scores = df["if_anomaly_score"].values
        ae_pct = np.searchsorted(self.ae_baseline, ae_errors) / len(self.ae_baseline) * 100
        if_pct = np.searchsorted(self.if_baseline, if_scores) / len(self.if_baseline) * 100

        # Vectorized risk band assignment
        sorted_items = sorted(self.percentile_thresholds.items(), key=lambda x: x[1])
        band_labels = [item[0] for item in sorted_items] # ["LOW","MEDIUM","HIGH","CRITICAL"]
        thresh_vals = [item[1] for item in sorted_items] # [80, 90, 95, 100]
        band_arr = np.array(band_labels)
        max_idx = len(band_labels) - 1

        # Composite risk band assignment
        ae_bands = band_arr[np.digitize(ae_pct, thresh_vals, right=True).clip(0, max_idx)]
        if_bands = band_arr[np.digitize(if_pct, thresh_vals, right=True).clip(0, max_idx)]
        composite = w1 * ae_pct + w2 * if_pct
        comp_bands = band_arr[np.digitize(composite, thresh_vals, right=True).clip(0, max_idx)]
        both_high = np.isin(ae_bands, ["HIGH", "CRITICAL"]) & np.isin(if_bands, ["HIGH", "CRITICAL"])

        # Vectorized top-K extraction
        contrib_cols = [c for c in df.columns if c.startswith("contribution_")]
        feat_names = [c.replace("contribution_", "") for c in contrib_cols]
        contrib_mat = df[contrib_cols].values.astype(float)   # shape (N, F)
        k = min(self.top_k, contrib_mat.shape[1])

        part_idx = np.argpartition(contrib_mat, -k, axis=1)[:, -k:]
        top_contributors_list = []
        for i in range(n):
            row_k_idx    = part_idx[i]
            sorted_k_idx = row_k_idx[np.argsort(contrib_mat[i, row_k_idx])[::-1]]
            top_contributors_list.append(
                [(feat_names[j], contrib_mat[i, j]) for j in sorted_k_idx]
            )

        # Classify each alert's threat scenario from its top contributing features after list is complete
        threat_categories = [classify_threat(tc) for tc in top_contributors_list]

        # Pre-extract z-score and delta arrays
        zscore_map = {}
        delta_map  = {}
        for c in df.columns:
            if c.endswith("_zscore"):
                zscore_map[c[:-len("_zscore")]] = df[c].values
            elif c.endswith("_rolling_delta"):
                delta_map[c[:-len("_rolling_delta")]] = df[c].values

        # Build explanation strings
        explanations = []
        for i in range(n):
            narrative_parts = []
            for feat, contrib_val in top_contributors_list[i]:
                if feat in zscore_map:
                    z = zscore_map[feat][i]
                    if not pd.isna(z):
                        delta = delta_map[feat][i] if feat in delta_map else None
                        delta_str = (
                            f", {delta:+.1f} vs 5-day avg"
                            if (delta is not None and not pd.isna(delta))
                            else ""
                        )
                        narrative_parts.append(f"{feat} (z={z:.1f}{delta_str}, contrib={contrib_val:.2f})")
                    else:
                        narrative_parts.append(f"{feat} ({contrib_val:.2f})")
                else:
                    narrative_parts.append(f"{feat} ({contrib_val:.2f})")

            explanations.append(
                f"AE deviation at {ae_pct[i]:.1f}th pct ({ae_bands[i]}); "
                f"IF anomaly at {if_pct[i]:.1f}th pct ({if_bands[i]}). "
                f"Composite: {comp_bands[i]} ({composite[i]:.1f}th pct). "
                f"Drivers: " + "; ".join(narrative_parts)
            )

        # Vectorized composite_priority 
        # Component 1: peak signal score (0-1)
        signal_scores = np.maximum(ae_pct, if_pct) / 100.0

        # Component 2: cross-channel flag density (0-1)
        present_flags = [f for f in CROSS_CHANNEL_FLAGS if f in df.columns]
        if present_flags:
            flag_counts = df[present_flags].apply(pd.to_numeric, errors="coerce").fillna(0).values.sum(axis=1)
            flag_scores = flag_counts / TOTAL_FLAGS
        else:
            flag_scores = np.zeros(n)

        # Component 3: role sensitivity (0-1)
        # np.where evaluates BOTH branches before masking, so np.vectorize would run on null
        # elements too — np.clip(None, ...) is numpy-version-dependent. Iterate explicitly instead.
        rs_raw = ldap_cols["role_sensitivity"]
        def _rs_score(v) -> float:
            if pd.isna(v):
                return 0.25
            if isinstance(v, str):
                return ROLE_SENSITIVITY_MAP.get(v.upper(), 0.25)
            return float(np.clip(float(v), 0.0, 1.0))
        sensitivity_scores = np.array([_rs_score(v) for v in rs_raw], dtype=float)

        # Component 4: sequence membership (binary; populated later, default 0)
        in_sequence = np.zeros(n, dtype=float)

        composite_priority = (
            0.50 * signal_scores
            + 0.20 * flag_scores
            + 0.15 * sensitivity_scores
            + 0.15 * in_sequence
        ) * 100.0

        # Assembling result DataFrame ---
        return pd.DataFrame({
            "alert_sequence_id":  np.full(n, None, dtype=object),
            "user":               df["user"].values,
            "day":                df["day"].values,
            "department":         ldap_cols["department"],
            "role":               ldap_cols["role"],
            "functional_unit":    ldap_cols["functional_unit"],
            "supervisor":         ldap_cols["supervisor"],
            "role_sensitivity":   ldap_cols["role_sensitivity"],
            "ae_percentile_rank": ae_pct,
            "ae_risk_band":       ae_bands,
            "top_contributors":   top_contributors_list,
            "threat_category":    threat_categories,
            "if_anomaly_score":   if_scores,
            "if_percentile_rank": if_pct,
            "if_risk_band":       if_bands,
            "composite_score":    composite,
            "composite_risk_band": comp_bands,
            "composite_priority": composite_priority,
            "both_signals_high":  both_high,
            "status":             np.full(n, "NEW", dtype=object),
            "explanation":        explanations,
        })


    def sort_alert_feed(self, alert_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the alert feed by risk band severity (CRITICAL first) and then by
        composite_priority descending within each band, so the most suspicious
        alerts surface at the top.

        Args:
            alert_df: Output of build_alert_df() or build_alert_from_row() calls.

        Returns:
            pd.DataFrame: Sorted alert DataFrame (new index, original data unchanged).
        """
        _severity = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sort_key = alert_df["composite_risk_band"].map(_severity).fillna(4)
        return (
            alert_df
            .assign(_band_rank=sort_key)
            .sort_values(["_band_rank", "composite_priority"], ascending=[True, False])
            .drop(columns=["_band_rank"])
            .reset_index(drop=True)
        )


    def validate_sequence_spans(self, alert_df: pd.DataFrame, max_span_days: int = 30) -> dict:
        """
        Checks that no alert sequence spans more than `max_span_days`.

        Groups rows by alert_sequence_id (non-null) and computes the span in
        calendar days between the earliest and latest alert in the group.

        Args:
            alert_df: DataFrame with alert_sequence_id and day columns.
            max_span_days: Maximum allowable calendar-day span per sequence.

        Returns:
            dict: {
                "passed": bool,
                "violations": pd.DataFrame with columns [alert_sequence_id, span_days],
                "max_span_days": int (configured limit),
            }
        """
        seq_df = alert_df[alert_df["alert_sequence_id"].notna()].copy()
        if seq_df.empty:
            return {"passed": True, "violations": pd.DataFrame(), "max_span_days": max_span_days}

        seq_df["day"] = pd.to_datetime(seq_df["day"])
        spans = (
            seq_df.groupby("alert_sequence_id")["day"]
            .agg(span_days=lambda s: (s.max() - s.min()).days)
            .reset_index()
        )
        violations = spans[spans["span_days"] > max_span_days]
        return {
            "passed": violations.empty,
            "violations": violations.reset_index(drop=True),
            "max_span_days": max_span_days,
        }


    def validate_priority_differentiation(self, alert_df: pd.DataFrame, target_r: float = 0.85) -> dict:
        """
        Verifies that composite_priority produces a meaningfully different sort order
        than raw peak percentile rank alone, restricted to elevated-risk rows
        (MEDIUM, HIGH, CRITICAL) where analyst triage actually happens.

        Running the check over all rows (including the LOW majority) inflates r
        because both signals trivially agree that LOW rows rank below elevated rows —
        that separation is noise for this test. Restricting to MEDIUM+ isolates the
        within-band reordering that the additional signals (flags, role sensitivity,
        sequence membership) are intended to produce.

        Args:
            alert_df: DataFrame with composite_priority, ae_percentile_rank,
                      if_percentile_rank, and composite_risk_band columns.
            target_r: Maximum acceptable Spearman r (default 0.85).

        Returns:
            dict: {
                "passed": bool  (True when r < target_r),
                "spearman_r": float,
                "p_value": float,
                "target_r": float,
                "n_rows": int   (number of elevated-risk rows evaluated),
            }
        """
        elevated = alert_df[
            alert_df["composite_risk_band"].isin({"MEDIUM", "HIGH", "CRITICAL"})
        ]

        if len(elevated) < 3:
            return {
                "passed": True,
                "spearman_r": None,
                "p_value": None,
                "target_r": target_r,
                "n_rows": len(elevated),
            }

        baseline = np.maximum(
            elevated["ae_percentile_rank"].values,
            elevated["if_percentile_rank"].values,
        )
        priority = elevated["composite_priority"].values

        r, p = spearmanr(baseline, priority)

        return {
            "passed": float(r) < target_r,
            "spearman_r": round(float(r), 4),
            "p_value": round(float(p), 6),
            "target_r": target_r,
            "n_rows": len(elevated),
        }


    def aggregate_alerts(self, alert_df: pd.DataFrame, window_days: int=7, min_risk: str="HIGH") -> pd.DataFrame:
        """
        Groups daily alerts into multi-day cases per user.
        
        Consecutive anomalous days within `window_days` of each other are merged into a single case, reducing alert
        fatigue and surfacing persistent threat patterns. Only rows at or above `min_risk` are considered
        when computing case boundaries.

        Args:
            alert_df: Output of `build_alert_df()`
            window_days: Maximum gap (in days) between alerts before a new case is opened.
                         A gap of exactly `window_days` keeps alerts in the same case.
            min_risk: Minimum composite_risk_band to include in case grouping
                      ("LOW", "MEDIUM", "HIGH", "CRITICAL").

        Returns:
            pd.DataFrame: One row per user-case, sorted by severity then peak composite score.
        """
        _severity = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        if min_risk not in _severity:
            raise ValueError(f"min_risk must be one of {list(_severity)}. Got: {min_risk!r}")

        df = alert_df.copy()
        df = df[df["composite_risk_band"].map(_severity) >= _severity[min_risk]].reset_index(drop=True)

        if df.empty:
            return pd.DataFrame(columns=[
                "user", "case_start", "case_end", "anomalous_days",
                "max_composite", "mean_composite", "peak_ae_percentile",
                "peak_if_percentile", "confirmed_days", "trend", "case_risk_band",
            ])

        df["day"] = pd.to_datetime(df["day"])
        df = df.sort_values(["user", "day"]).reset_index(drop=True)

        # Identify case breaks (gap > window_days)
        day_diff = df.groupby("user")["day"].diff().dt.days.fillna(window_days + 1)
        new_case_flag = day_diff > window_days
        df["_case_id"] = new_case_flag.groupby(df["user"]).cumsum()

        # Per-case aggregation
        agg = df.groupby(["user", "_case_id"]).agg(
            case_start = ("day", "min"),
            case_end = ("day", "max"),
            anomalous_days = ("day", "count"),
            max_composite = ("composite_score", "max"),
            mean_composite = ("composite_score", "mean"),
            peak_ae_percentile = ("ae_percentile_rank", "max"),
            peak_if_percentile = ("if_percentile_rank", "max"),
            confirmed_days = ("both_signals_high", "sum"),
        ).reset_index()

        # First-half vs second-half mean composite within the case
        trends = (
            df.groupby(["user", "_case_id"])["composite_score"]
            .apply(lambda s: (
                "SINGLE_DAY"    if len(s) <= 1
                else "ESCALATING"   if s.iloc[len(s) // 2:].mean() > s.iloc[:len(s) // 2].mean()
                else "DE-ESCALATING"
            ))
            .reset_index(name="trend")
        )

        agg = (
            agg.merge(trends, on=["user", "_case_id"])
               .drop(columns=["_case_id"])
        )

        # Case-level risk band derived from peak composite score
        agg["case_risk_band"] = agg["max_composite"].apply(self.assign_risk_band)

        # Sorting by risk-band and then by peak score
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        agg["_sort_key"] = agg["case_risk_band"].map(severity_order)
        agg = (
            agg.sort_values(["_sort_key", "max_composite"], ascending=[True, False])
               .drop(columns=["_sort_key"])
               .reset_index(drop=True)
        )

        return agg


def save_table(df: pd.DataFrame, save_path: str) -> None:
    """
    Saves the DataFrame table as a CSV or Parquet file to the specified path.

    Args:
        df: DataFrame table
        save_path: Path to save table to

    Returns:
        None:
    """
    # Ensuring output directory exists
    output_dir = os.path.join("explainability", "alert_table")
    os.makedirs(output_dir, exist_ok=True)

    format = save_path.split(".")[-1]
    if format not in ("csv", "parquet"):
        raise ValueError(f"Please specify either 'csv' or 'parquet' format. Got {format}.")
    
    full_path = os.path.join(output_dir, save_path)
    if format == "csv":
        df.to_csv(full_path, index=False)
    else:
        # Converting object columns to JSON for parquet
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == "object":
                df_copy[col] = df_copy[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )
        df_copy.to_parquet(full_path, index=False)

    print("Successfully saved to:", full_path)
