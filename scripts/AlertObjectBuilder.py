import os
import numpy as np
import pandas as pd


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
        
    
    def extract_top_contributors(self, row: pd.Series) -> list:
        """
        Extracts the top-K contributing factors from a row.
        
        Args:
            row: A row consisting of metadata and recontruction error data
            
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
        
        
    def build_alert_from_row(self, row: pd.Series) -> dict:
        """
        Builds an alert dictionary for a single sample using both AE and IF signals.
        
        Args:
            row: A row consisting of metadata, reconstruction error data, and IF anomaly score
            
        Returns:
            dict: A structured alert object
        """
        # Computing AE percentile and risk band
        ae_error = row["total_reconstruction_error"]
        ae_percentile = self.compute_ae_percentile(ae_error)
        ae_risk_band = self.assign_risk_band(ae_percentile)
        top_features = self.extract_top_contributors(row)
        
        # Computing IF percentile and risk band
        if_score = row["if_anomaly_score"]
        if_percentile = self.compute_if_percentile(if_score)
        if_risk_band = self.assign_risk_band(if_percentile)
        
        # Creating explanation statement
        explanation = (
            f"AE deviation at {ae_percentile:.2f}th percentile ({ae_risk_band}); "
            f"IF anomaly at {if_percentile:.2f}th percentile ({if_risk_band}). "
            f"Top features: " + ", ".join([f"{feat} ({val:.2f})" for feat, val in top_features])
        )
        
        # Creating alert dictionary
        alert = {
            "user": row["user"],
            "pc": row["pc"],
            "day": row["day"],
            "ae_percentile_rank": ae_percentile,
            "ae_risk_band": ae_risk_band,
            "top_contributors": top_features,
            "if_anomaly_score": if_score,
            "if_percentile_rank": if_percentile,
            "if_risk_band": if_risk_band,
            "explanation": explanation
        }
        
        return alert
        
        
    def build_alert_df(self, explanation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates an alert DataFrame from an aggregated table consisting of reconstruction errors
        and anomaly scores.
        
        Args:
            explanation_df: The enriched DataFrame containing AE reconstruction errors and IF anomaly scores
            
        Returns:
            pd.DataFrame: An alert-ready structured DataFrame
        """
        alerts = []
        
        for _, row in explanation_df.iterrows():
            alert = self.build_alert_from_row(row)
            alerts.append(alert)
            
        return pd.DataFrame(alerts)