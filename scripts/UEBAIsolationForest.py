import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score


class UEBAIsolationForest:
    """
    Isolation Forest for anomaly detection on Autoencoder latent embeddings.
    """
    
    def __init__(self, n_estimators: int=200, max_samples: str="auto", contamination: float=0.001, random_state: int=42) -> None:
        """
        Initializes the Isolation Forest.
        
        Args:
            n_estimators: The number of trees in the forest
            max_samples: The subsample size for each tree
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
            
        Returns:
            None:
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
    
     
    def train(self, latent_embeddings: np.ndarray):
        """
        Trains the Isolation Forest on latent embeddings.
        
        Args:
            latent_embeddings: Latent embedding matrix of shape: (n_samples, latent_emb_dim)
            
        Returns:
            A history object of the model's training
        """
        history = self.model.fit(latent_embeddings)
        return history
    
    
    def anomaly_score(self, latent_embeddings: np.ndarray) -> np.ndarray:
        """
        Computes the anomaly scores for latent embeddings, where a higher score signifies more
        anomalous activity.
        
        Args:
            latent_embeddings: The latent embeddings matrix of shape: (n_samples, latent_emb_dim)
            
        Returns:
            np.ndarray: Anomaly scores
        """
        scores = -self.model.score_samples(latent_embeddings)
        return scores
    
    
    def predict(self, latent_embeddings: np.ndarray) -> np.ndarray:
        """
        Predicts anomaly labels using model threshold, where -1 signifies anomalous activity and
        1 signifies normal behavior.
        
        Args:
            latent_embeddings: The latent embedding matrix.
            
        Returns:
            np.ndarray: Binary predictions conveying normal or anomalous
        """
        labels = self.model.predict(latent_embeddings)
        return labels
    
    
    def save(self, save_path: str) -> None:
        """
        Save the trained Isolation Forest model.
        
        Args:
            save_path: The path where the Isolation Forest will be saved
            
        Returns:
            None:
        """
        joblib.dump(self.model, save_path)
        
    
    def load(self, load_path: str) -> None:
        """
        Loads a previously trained Isolation Forest model.
        
        Args:
            load_path: File path from where to load the pretrained model
            
        Returns:
            None:
        """
        self.model = joblib.load(load_path)


def compute_contamination_rate(total_embeddings: np.ndarray, normal_embeddings: np.ndarray) -> float:
    """
    Calculates the optimal contamination rate for an Isolation Forest.
    
    The normal embeddings are subtracted from the total embeddings to identify the ratio of insiders.
    The contamination rate is then assigned using `max(computed_value, 0.001)` for numerical stability.
    
    Args:
        total_embeddings: Latent embeddings containing normal and insider rows
        normal_embeddings: Latent embeddings containing only normal-behavior rows
        
    Returns:
        float: The optimal contamination rate
    """
    n_insiders = len(total_embeddings) - len(normal_embeddings)
    computed_rate = n_insiders / len(total_embeddings)
    
    return max(computed_rate, 0.001)


def compute_separation_ratio(anomaly_scores: np.ndarray, insider_mask: pd.Series | np.ndarray) -> float:
    """
    Measures how well the trained Isolation Forest distinguishes between normal and anomalous
    behavior.
    
    A ratio > 1.0 signifies the model is performing well. A ration <= 1.0 signifies the model
    may be struggling to distinguish between normal and insider behavior. Ratio is computed
    using `mean_anomaly_score / mean_normal_score`.
    
    Args:
        anomaly_scores: Computed anomaly scores
        insider_mask: Array denoting which rows are truly normal or anomalous
        
    Returns:
        float: Separation ratio
    """
    if isinstance(insider_mask, pd.Series):
        insider_mask = insider_mask.values.astype(bool)
        
    # Computing means
    mean_anomaly_score = anomaly_scores[insider_mask].mean()
    mean_normal_score = anomaly_scores[~insider_mask].mean()
    
    # Deriving ratio
    ratio = mean_anomaly_score / mean_normal_score
    
    print(f"Mean Insider Score: {mean_anomaly_score:.4f}")
    print(f"Mean Normal Score: {mean_normal_score:.4f}")
    print(f"Separation Ratio: {ratio:.2f}x")
    return ratio


def compute_roc_auc_score(insider_labels: pd.Series | np.ndarray, anomaly_scores: np.ndarray) -> float:
    """
    Compute the ROC AUC score based on the provided insider labels and anomaly scores.
    
    Args:
        insider_labels: Array denoting which rows consist of insider activity
        anomaly_scores: Scores generated by IF
    
    Returns:
        float: ROC AUC score
    """
    if isinstance(insider_labels, pd.Series):
        insider_labels = insider_labels.astype(int).values
    
    score = roc_auc_score(insider_labels, anomaly_scores)
    print(f"AUROC: {score:.4f}")
    return score


def compute_avg_prec_score(insider_labels: pd.Series | np.ndarray, anomaly_scores: np.ndarray) -> float:
    """
    Compute the average precision score based on the provided insider labels and anomaly scores.
    
    Args:
        insider_labels: Array denoting which rows consist of insider activity
        anomaly_scores: Scores generated by IF
    
    Returns:
        float: Average precision score
    """
    if isinstance(insider_labels, pd.Series):
        insider_labels = insider_labels.astype(int).values
        
    score = average_precision_score(insider_labels, anomaly_scores)
    print(f"AUPRC: {score:.4f}")
    return score


def compute_recall_thresholds(insider_labels: pd.Series | np.ndarray, anomaly_scores: np.ndarray, percentiles: list=[80, 90, 95]) -> dict[str, float]:
    """
    Compute the recall at several percentile thresholds.
    
    Args:
        insider_labels: Array denoting which rows consist of insider activity
        percentiles: List of percentile thresholds
        anomaly_scores: Score generate by IF
        
    Returns:
        dict: Recall values {percentile: recall score}
    """
    values = {}
    
    if isinstance(insider_labels, pd.Series):
        insider_labels = insider_labels.astype(int).values
    
    for pct in percentiles:
        threshold = np.percentile(anomaly_scores, pct)
        flagged = anomaly_scores >= threshold
        captured = insider_labels[flagged].sum()
        total_insiders = insider_labels.sum()
        recall = captured/total_insiders
        values[str(pct)] = recall
        print(f"Recall at {pct}th Percentile: {recall:.3f} ({captured} of {total_insiders} insider-days captured)")
        
    return values