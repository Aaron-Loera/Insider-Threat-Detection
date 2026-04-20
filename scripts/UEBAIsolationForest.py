import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


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


def compute_separation_ratio(anomaly_scores: np.ndarray, insider_mask: pd.Series | np.ndarray, save_path: str | None = None) -> float:
    """
    Measures how well the trained Isolation Forest distinguishes between normal and anomalous
    behavior.

    A ratio > 1.0 signifies the model is performing well. A ration <= 1.0 signifies the model
    may be struggling to distinguish between normal and insider behavior. Ratio is computed
    using `mean_anomaly_score / mean_normal_score`. Renders overlaid histograms of normal vs
    insider scores with dashed mean lines.

    Args:
        anomaly_scores: Computed anomaly scores
        insider_mask: Array denoting which rows are truly normal or anomalous
        save_path: Optional path to persist the figure as PNG

    Returns:
        float: Separation ratio
    """
    if isinstance(insider_mask, pd.Series):
        insider_mask = insider_mask.values.astype(bool)

    mean_anomaly_score = anomaly_scores[insider_mask].mean()
    mean_normal_score = anomaly_scores[~insider_mask].mean()
    ratio = mean_anomaly_score / mean_normal_score

    plt.figure(figsize=(10, 5))
    plt.hist(anomaly_scores[~insider_mask], bins=50, color="steelblue", alpha=0.6,
             label=f"Normal (mean={mean_normal_score:.4f})")
    plt.hist(anomaly_scores[insider_mask], bins=50, color="coral", alpha=0.6,
             label=f"Insider (mean={mean_anomaly_score:.4f})")
    plt.axvline(mean_normal_score, color="steelblue", linestyle="--", alpha=0.9)
    plt.axvline(mean_anomaly_score, color="coral", linestyle="--", alpha=0.9)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title(f"Anomaly Score Distribution (Separation Ratio: {ratio:.2f}x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "separation_ratio.png"), dpi=150)
    plt.show()

    return ratio


def compute_roc_auc_score(insider_labels: pd.Series | np.ndarray, anomaly_scores: np.ndarray, save_path: str | None = None) -> float:
    """
    Compute the ROC AUC score based on the provided insider labels and anomaly scores.
    Renders the ROC curve with a random-guess diagonal reference.

    Args:
        insider_labels: Array denoting which rows consist of insider activity
        anomaly_scores: Scores generated by IF
        save_path: Optional path to persist the figure as PNG

    Returns:
        float: ROC AUC score
    """
    if isinstance(insider_labels, pd.Series):
        insider_labels = insider_labels.astype(int).values

    score = roc_auc_score(insider_labels, anomaly_scores)
    fpr, tpr, _ = roc_curve(insider_labels, anomaly_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color="steelblue", label=f"ROC (AUROC = {score:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUROC = {score:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "roc_curve.png"), dpi=150)
    plt.show()

    return score


def compute_avg_prec_score(insider_labels: pd.Series | np.ndarray, anomaly_scores: np.ndarray, save_path: str | None = None) -> float:
    """
    Compute the average precision score based on the provided insider labels and anomaly scores.
    Renders the Precision-Recall curve with a baseline at the positive-class prevalence.

    Args:
        insider_labels: Array denoting which rows consist of insider activity
        anomaly_scores: Scores generated by IF
        save_path: Optional path to persist the figure as PNG

    Returns:
        float: Average precision score
    """
    if isinstance(insider_labels, pd.Series):
        insider_labels = insider_labels.astype(int).values

    score = average_precision_score(insider_labels, anomaly_scores)
    precision, recall, _ = precision_recall_curve(insider_labels, anomaly_scores)
    base_rate = insider_labels.mean()

    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, color="steelblue", label=f"PR (AUPRC = {score:.4f})")
    plt.axhline(base_rate, color="gray", linestyle="--", alpha=0.7,
                label=f"Baseline (class balance = {base_rate:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AUPRC = {score:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "pr_curve.png"), dpi=150)
    plt.show()

    return score


def compute_recall_thresholds(insider_labels: pd.Series | np.ndarray, anomaly_scores: np.ndarray, percentiles: list=[80, 90, 95], save_path: str | None = None) -> dict[str, float]:
    """
    Compute the recall at several percentile thresholds. Renders a bar chart of recall per
    percentile with captured/total annotations above each bar.

    Args:
        insider_labels: Array denoting which rows consist of insider activity
        anomaly_scores: Score generate by IF
        percentiles: List of percentile thresholds
        save_path: Optional path to persist the figure as PNG

    Returns:
        dict: Recall values {percentile: recall score}
    """
    values = {}
    captured_per_pct = {}

    if isinstance(insider_labels, pd.Series):
        insider_labels = insider_labels.astype(int).values

    total_insiders = insider_labels.sum()

    for pct in percentiles:
        threshold = np.percentile(anomaly_scores, pct)
        flagged = anomaly_scores >= threshold
        captured = insider_labels[flagged].sum()
        recall = captured / total_insiders
        values[str(pct)] = recall
        captured_per_pct[pct] = captured

    labels = [f"{pct}th" for pct in percentiles]
    recalls = [values[str(pct)] for pct in percentiles]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, recalls, color="steelblue", alpha=0.85)
    for bar, pct in zip(bars, percentiles):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{captured_per_pct[pct]}/{total_insiders}",
            ha="center",
            va="bottom",
        )
    plt.xlabel("Percentile Threshold")
    plt.ylabel("Recall")
    plt.title("Recall at Percentile Thresholds")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "recall_percentiles.png"), dpi=150)
    plt.show()

    return values