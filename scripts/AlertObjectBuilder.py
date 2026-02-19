import os
import numpy as np
import pandas as pd


class AlertObjectBuilder:
    """
    Converts raw reconstruction error outputs into contextual, SOC-ready alert objects. Can operat on Autoencoder errors,
    Isolation Forest scores, or hybridd risk metrics.
    """
    
    def __init__(self, percentile_thresholds: dict | None=None, top_k: int=3) -> None:
        """
        Initializing the alert object builder.
        
        Args:
            percentile_thresholds: The risk band thresholds
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
        self.baseline_errors = None
        
    
    def fit_baseline(self, reconstruction_errors: np.ndarray) -> None:
        """
        Stores the baseline reconstruction error distribution.
        
        Args:
            reconstruction_errors: Array of baseline reconstruction errors.
            
        Returns:
            None:
        """
        self.baseline_errors = np.sort(reconstruction_errors)
        
    
    def compute_percentile(self, error: float) -> float:
        """
        Computes the percentile rank of an error value.
        
        Args:
            error: The error value to rank
            
        Returns:
            float: The percentile the error value falls within (0-100)
        """
        if self.baseline_errors is None:
            raise ValueError("Baseline distribution not fitted.")
        
        # Calculating the percentile
        percentile = (np.searchsorted(self.baseline_errors, error)) / len(self.baseline_errors) * 100
        return percentile
    
    
    def assign_risk_band(self, percentile: float) -> str:
        """
        Assigns risk bands based on the provided percentile.
        
        Args:
            percentile: The assigned percentile of an error value
            
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
        Builds an alert dictionary for a single sample.
        
        Args:
            row: A row consisting of metadata and recontruction error data
            
        Returns:
            dict: A structured alert object
        """
        # Creating percential assignment and risk band
        error = row["total_reconstruction_error"]
        percentile = self.compute_percentile(error)
        risk_band = self.assign_risk_band(percentile)
        top_features = self.extract_top_contributors(row)
        
        # Creating explanation statement
        explanation = (
            f"Behavior falls in the {percentile:.2f}th percentile of reconstruction deviation. Primary contributors: "\
            + ", ".join([f"{feat} ({val:.2f})" for feat, val in top_features])
        )
        
        # Creating alert dictionary
        alert = {
            "user": row["user"],
            "pc": row["pc"],
            "day": row["day"],
            "percentile_rank": percentile,
            "risk_band": risk_band,
            "top_contributors": top_features,
            "explanation": explanation
        }
        
        return alert
        
        
    def build_alert_df(self, explanation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates an alert DataFrame from a reconstruction error explanation DataFrame.
        
        Args:
            explanation_df: The reconstuction error explanation DataFrame
            
        Returns:
            pd.DataFrame: An alert-ready structured DataFrame
        """
        alerts = []
        
        for _, row in explanation_df.iterrows():
            alert = self.build_alert_from_row(row)
            alerts.append(alert)
            
        return pd.DataFrame(alerts)