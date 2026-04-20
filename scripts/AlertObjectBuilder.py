import os
import json
import numpy as np
import pandas as pd


class AlertObjectBuilder:
    """
    Converts raw reconstruction error outputs into contextual, SOC-ready alert objects.
    Operates on both Autoencoder reconstruction errors and Isolation Forest anomaly scores,
    producing a unified alert with per-signal percentile ranks and risk bands.
    """

    def __init__(
        self,
        percentile_thresholds: dict | None = None,
        top_k: int = 3,
        ae_absolute_thresholds: dict | None = None,
        if_absolute_thresholds: dict | None = None,
    ) -> None:
        """
        Initializing the alert object builder.

        Args:
            percentile_thresholds: Legacy percentile-based risk band thresholds. Used for banding
                only when ae_absolute_thresholds / if_absolute_thresholds are not provided.
            top_k: Number of top contributing features to extract
            ae_absolute_thresholds: Absolute AE reconstruction-error thresholds calibrated against
                a held-out clean population (e.g. {"LOW": t_low, "MEDIUM": t_med, "HIGH": t_hi,
                "CRITICAL": inf}). When set, these replace percentile_thresholds for AE banding.
            if_absolute_thresholds: Same as ae_absolute_thresholds but for IF anomaly scores.

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
        self.ae_absolute_thresholds = ae_absolute_thresholds
        self.if_absolute_thresholds = if_absolute_thresholds


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


    def assign_risk_band_from_score(self, score: float, absolute_thresholds: dict) -> str:
        """
        Assigns a risk band by comparing a raw score against calibrated absolute thresholds.

        Args:
            score: Raw AE reconstruction error or IF anomaly score
            absolute_thresholds: Ordered dict {"LOW": t_low, ..., "CRITICAL": inf}

        Returns:
            str: The risk band label
        """
        for label, thresh in absolute_thresholds.items():
            if thresh is None or score <= thresh:
                return label
        return "CRITICAL"


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
                 richer explanation text.
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
        if self.ae_absolute_thresholds is not None:
            ae_risk_band = self.assign_risk_band_from_score(ae_error, self.ae_absolute_thresholds)
        else:
            ae_risk_band = self.assign_risk_band(ae_percentile)
        top_features = self.extract_top_contributors(row)

        # Computing IF percentile and risk band
        if_score = row["if_anomaly_score"]
        if_percentile = self.compute_if_percentile(if_score)
        if self.if_absolute_thresholds is not None:
            if_risk_band = self.assign_risk_band_from_score(if_score, self.if_absolute_thresholds)
        else:
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

        # Creating alert dictionary
        alert = {
            "user": row["user"],
            "day": row["day"],
            "ae_percentile_rank": ae_percentile,
            "ae_risk_band": ae_risk_band,
            "top_contributors": top_features,
            "if_anomaly_score": if_score,
            "if_percentile_rank": if_percentile,
            "if_risk_band": if_risk_band,
            "composite_score": composite_score,
            "composite_risk_band": composite_risk_band,
            "both_signals_high": both_signals_high,
            "explanation": explanation
        }

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

        # Vectorized percentile computation
        ae_errors = df["total_reconstruction_error"].values
        if_scores = df["if_anomaly_score"].values
        ae_pct = np.searchsorted(self.ae_baseline, ae_errors) / len(self.ae_baseline) * 100
        if_pct = np.searchsorted(self.if_baseline, if_scores) / len(self.if_baseline) * 100

        # Vectorized risk band assignment
        # Composite always uses percentile thresholds (percentile-space fusion score)
        sorted_pct_items = sorted(self.percentile_thresholds.items(), key=lambda x: x[1])
        pct_band_labels = [item[0] for item in sorted_pct_items]
        pct_thresh_vals = [item[1] for item in sorted_pct_items]
        pct_band_arr = np.array(pct_band_labels)
        max_idx = len(pct_band_labels) - 1

        composite = w1 * ae_pct + w2 * if_pct
        comp_bands = pct_band_arr[np.digitize(composite, pct_thresh_vals, right=True).clip(0, max_idx)]

        # AE and IF bands: use absolute thresholds when calibrated, else fall back to percentile
        if self.ae_absolute_thresholds is not None:
            abs_items = sorted(self.ae_absolute_thresholds.items(),
                               key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))
            abs_labels = np.array([item[0] for item in abs_items])
            abs_vals = [float("inf") if item[1] is None else item[1] for item in abs_items]
            ae_bands = abs_labels[np.digitize(ae_errors, abs_vals, right=True).clip(0, max_idx)]
        else:
            ae_bands = pct_band_arr[np.digitize(ae_pct, pct_thresh_vals, right=True).clip(0, max_idx)]

        if self.if_absolute_thresholds is not None:
            abs_items = sorted(self.if_absolute_thresholds.items(),
                               key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))
            abs_labels = np.array([item[0] for item in abs_items])
            abs_vals = [float("inf") if item[1] is None else item[1] for item in abs_items]
            if_bands = abs_labels[np.digitize(if_scores, abs_vals, right=True).clip(0, max_idx)]
        else:
            if_bands = pct_band_arr[np.digitize(if_pct, pct_thresh_vals, right=True).clip(0, max_idx)]
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

        # Assembling result DataFrame ---
        return pd.DataFrame({
            "user":               df["user"].values,
            "day":                df["day"].values,
            "ae_percentile_rank": ae_pct,
            "ae_risk_band":       ae_bands,
            "top_contributors":   top_contributors_list,
            "if_anomaly_score":   if_scores,
            "if_percentile_rank": if_pct,
            "if_risk_band":       if_bands,
            "composite_score":    composite,
            "composite_risk_band": comp_bands,
            "both_signals_high":  both_high,
            "explanation":        explanations,
        })


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
    output_dir = r"explainability\alert_table"
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
