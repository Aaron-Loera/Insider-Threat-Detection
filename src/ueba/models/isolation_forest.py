import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


class UEBAIsolationForest:
    """
    Isolation Forest for anomaly detection on Autoencoder latent embeddings.
    """

    def __init__(self, n_estimators: int=200, max_samples: str="auto", contamination: str | float="auto", random_state: int=42) -> None:
        """
        Initializes the Isolation Forest.

        Args:
            n_estimators: The number of trees in the forest
            max_samples: The subsample size for each tree
            contamination: Expected proportion of anomalies in the training data. Use "auto"
                when training on insider-free data and deriving alert thresholds from a
                separate clean calibration baseline.
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


    @staticmethod
    def compute_contamination_rate(total_embeddings: np.ndarray, normal_embeddings: np.ndarray) -> str | float:
        """
        Returns the contamination value to pass to sklearn's IsolationForest.

        Since training is done on insider-free embeddings, there is no meaningful insider
        ratio to compute in the training set itself. sklearn's "auto" heuristic
        (offset_ = -0.5) avoids hard-coding an operational alert rate into the model fit;
        thresholds for alerting should instead be derived from the held-out clean
        calibration baseline (see UEBA_CALIBRATION_PATH / IF_BASELINE_PATH).

        Args:
            total_embeddings: Latent embeddings containing normal and insider rows (unused)
            normal_embeddings: Latent embeddings containing only normal-behavior rows (unused)

        Returns:
            str: "auto"
        """
        return "auto"


    def compute_separation_ratio(self, latent_embeddings: np.ndarray, insider_labels: pd.Series | np.ndarray, save_path: str | None = None) -> float:
        """
        Measures how well the trained Isolation Forest distinguishes between normal and anomalous
        behavior.

        A ratio > 1.0 signifies the model is performing well. A ratio <= 1.0 signifies the model
        may be struggling to distinguish between normal and insider behavior. Ratio is computed
        using `mean_anomaly_score / mean_normal_score`. Renders overlaid histograms of normal vs
        insider scores with dashed mean lines.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim)
            insider_labels: Binary labels (1 = insider, 0 = normal)
            save_path: Optional path to persist the figure as PNG

        Returns:
            float: Separation ratio
        """
        scores = self.anomaly_score(latent_embeddings)

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        mean_anomaly_score = scores[insider_labels == 1].mean()
        mean_normal_score = scores[insider_labels == 0].mean()
        ratio = mean_anomaly_score / mean_normal_score

        plt.figure(figsize=(10, 5))
        plt.hist(scores[insider_labels == 0], bins=50, color="steelblue", alpha=0.6,
                 label=f"Normal (mean={mean_normal_score:.4f})")
        plt.hist(scores[insider_labels == 1], bins=50, color="coral", alpha=0.6,
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


    def compute_roc_auc_score(self, latent_embeddings: np.ndarray, insider_labels: pd.Series | np.ndarray, save_path: str | None = None) -> float:
        """
        Compute the ROC AUC score for the Isolation Forest. Renders the ROC curve with a
        random-guess diagonal reference.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim)
            insider_labels: Binary labels (1 = insider, 0 = normal)
            save_path: Optional path to persist the figure as PNG

        Returns:
            float: ROC AUC score
        """
        scores = self.anomaly_score(latent_embeddings)

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        score = roc_auc_score(insider_labels, scores)
        fpr, tpr, _ = roc_curve(insider_labels, scores)

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


    def compute_avg_prec_score(self, latent_embeddings: np.ndarray, insider_labels: pd.Series | np.ndarray, save_path: str | None = None) -> float:
        """
        Compute the average precision score for the Isolation Forest. Renders the Precision-Recall
        curve with a baseline at the positive-class prevalence.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim)
            insider_labels: Binary labels (1 = insider, 0 = normal)
            save_path: Optional path to persist the figure as PNG

        Returns:
            float: Average precision score
        """
        scores = self.anomaly_score(latent_embeddings)

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        score = average_precision_score(insider_labels, scores)
        precision, recall, _ = precision_recall_curve(insider_labels, scores)
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


    def compute_recall_thresholds(self, latent_embeddings: np.ndarray, insider_labels: pd.Series | np.ndarray, percentiles: list = [80, 90, 95], threshold_source: np.ndarray | None = None, save_path: str | None = None) -> dict[str, float]:
        """
        Compute the recall at several percentile thresholds of the anomaly score distribution.
        Renders a bar chart of recall per percentile with captured/total annotations above each bar.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim)
            insider_labels: Binary labels (1 = insider, 0 = normal)
            percentiles: List of percentile thresholds
            threshold_source: Optional 1-D score array used to derive percentile thresholds. When
                supplied, thresholds are computed from this distribution (e.g. the insider-free
                calibration baseline) rather than the evaluation scores themselves — avoiding the
                circularity of measuring recall against a threshold defined by the same data.
            save_path: Optional path to persist the figure as PNG

        Returns:
            dict: Recall values {percentile: recall score}
        """
        scores = self.anomaly_score(latent_embeddings)
        source = np.asarray(threshold_source) if threshold_source is not None else scores

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        values = {}
        captured_per_pct = {}
        total_insiders = insider_labels.sum()

        for pct in percentiles:
            threshold = np.percentile(source, pct)
            flagged = scores >= threshold
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


    def compute_confusion_matrix(self, latent_embeddings: np.ndarray, insider_labels: pd.Series | np.ndarray, train_score_dist: np.ndarray, percentiles: list = [80, 90, 95], save_path: str | None = None) -> dict[str, dict[str, float]]:
        """
        Computes confusion matrix components at operational percentile thresholds derived from
        the training score distribution. Renders a grouped bar chart of TP and FP counts per threshold.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim)
            insider_labels: Binary labels (1 = insider, 0 = normal)
            train_score_dist: Anomaly scores from the training distribution for threshold derivation
            percentiles: List of percentile thresholds
            save_path: Optional path to persist the figure as PNG

        Returns:
            dict: {percentile_str: {"TN": int, "FP": int, "FN": int, "TP": int,
                   "precision": float, "recall": float, "fpr": float}}
        """
        scores = self.anomaly_score(latent_embeddings)

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        thresholds = {p: float(np.percentile(train_score_dist, p)) for p in percentiles}
        results = {}

        print(f"\n{'Threshold':>12}  {'TN':>8}  {'FP':>8}  {'FN':>8}  {'TP':>8}  {'Precision':>10}  {'Recall':>10}  {'FPR':>10}")
        print("-" * 85)

        for pct, thresh in thresholds.items():
            predicted = (scores >= thresh).astype(int)
            cm = confusion_matrix(insider_labels, predicted, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            results[str(pct)] = {
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                "precision": precision, "recall": recall, "fpr": fpr,
            }
            print(f"{pct}th pct:  {tn:>8}  {fp:>8}  {fn:>8}  {tp:>8}  {precision:>10.4f}  {recall:>10.4f}  {fpr:>10.4f}")

        labels = [f"{p}th" for p in percentiles]
        tps = [results[str(p)]["TP"] for p in percentiles]
        fps = [results[str(p)]["FP"] for p in percentiles]
        x_pos = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x_pos - width / 2, tps, width, label="True Positives", color="#e84545", alpha=0.85)
        plt.bar(x_pos + width / 2, fps, width, label="False Positives", color="#3a86a8", alpha=0.85)
        for i, (tp_val, fp_val) in enumerate(zip(tps, fps)):
            plt.text(x_pos[i] - width / 2, tp_val + 0.3, str(tp_val), ha="center", va="bottom", fontsize=9)
            plt.text(x_pos[i] + width / 2, fp_val + 0.3, str(fp_val), ha="center", va="bottom", fontsize=9)
        plt.xticks(x_pos, labels)
        plt.xlabel("Percentile Threshold")
        plt.ylabel("Count")
        plt.title("Confusion Matrix: TP vs FP at Operational Thresholds")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=150)
        plt.show()

        return results


    def compute_precision_at_recall(self, latent_embeddings: np.ndarray, insider_labels: pd.Series | np.ndarray, target_recalls: list = [0.22, 0.50, 0.80], save_path: str | None = None) -> dict[str, float]:
        """
        Computes best achievable precision at specified recall levels. Renders the PR curve
        with vertical lines annotating each target recall level.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim)
            insider_labels: Binary labels (1 = insider, 0 = normal)
            target_recalls: List of target recall levels to evaluate precision at
            save_path: Optional path to persist the figure as PNG

        Returns:
            dict: {str(target_recall): best_precision_achieved}
        """
        scores = self.anomaly_score(latent_embeddings)

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        prec_vals, rec_vals, _ = precision_recall_curve(insider_labels, scores)
        base_rate = insider_labels.mean()

        results = {}
        print(f"\n{'Target Recall':>15}  {'Achieved Precision':>20}")
        print("-" * 38)
        for target in target_recalls:
            mask = rec_vals >= target
            best_prec = float(prec_vals[mask].max()) if mask.any() else 0.0
            results[str(target)] = best_prec
            print(f"{target:>15.2f}  {best_prec:>20.4f}")

        line_colors = ["#e84545", "#d4a017", "#3a86a8"]
        plt.figure(figsize=(10, 5))
        plt.plot(rec_vals, prec_vals, color="steelblue", label=f"PR Curve (base rate = {base_rate:.4f})")
        plt.axhline(base_rate, color="gray", linestyle="--", alpha=0.7, label=f"Baseline ({base_rate:.4f})")
        for target, color in zip(target_recalls, line_colors):
            prec = results[str(target)]
            plt.axvline(target, color=color, linestyle="--", alpha=0.8,
                        label=f"Recall={target:.2f} → Precision={prec:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision at Target Recall Levels")
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "precision_at_recall.png"), dpi=150)
        plt.show()

        return results


    def compute_user_detection_rate(self, latent_embeddings: np.ndarray, test_df: pd.DataFrame, insiders_df: pd.DataFrame, train_score_dist: np.ndarray, percentiles: list = [80, 90, 95], save_path: str | None = None) -> dict[str, dict]:
        """
        Computes user-level detection rate: what fraction of distinct insider users have at least
        one day flagged above each operational threshold. Renders a bar chart of detection rates
        per percentile.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim); rows aligned with test_df
            test_df: DataFrame with columns 'user' and 'day'; rows aligned with latent_embeddings
            insiders_df: Insider metadata with columns 'user', 'start_day', 'end_day'
            train_score_dist: Anomaly scores from the training distribution for threshold derivation
            percentiles: List of percentile thresholds
            save_path: Optional path to persist the figure as PNG

        Returns:
            dict: {percentile_str: {"detected": int, "total_insiders": int, "rate": float}}
        """
        scores = self.anomaly_score(latent_embeddings)

        thresholds = {p: float(np.percentile(train_score_dist, p)) for p in percentiles}

        scored_df = test_df[["user", "day"]].copy()
        scored_df["score"] = scores
        scored_df["day"] = pd.to_datetime(scored_df["day"])
        scored_df["user"] = scored_df["user"].str.strip().str.lower()

        user_peak = scored_df.groupby("user")["score"].max()

        test_start = scored_df["day"].min()
        test_end = scored_df["day"].max()

        insiders_df = insiders_df.copy()
        insiders_df["start_day"] = pd.to_datetime(insiders_df["start_day"])
        insiders_df["end_day"] = pd.to_datetime(insiders_df["end_day"])
        insiders_df["user_key"] = insiders_df["user"].str.strip().str.lower()

        active_mask = (insiders_df["start_day"] <= test_end) & (insiders_df["end_day"] >= test_start)
        active_insider_users = set(insiders_df.loc[active_mask, "user_key"])
        total_insiders = len(active_insider_users)

        results = {}
        print(f"\n{'Percentile':>12}  {'Detected':>10}  {'Total Insiders':>16}  {'Detection Rate':>16}")
        print("-" * 58)

        for pct, thresh in thresholds.items():
            flagged_users = set(user_peak[user_peak >= thresh].index.str.strip().str.lower())
            detected = len(active_insider_users & flagged_users)
            rate = detected / total_insiders if total_insiders > 0 else 0.0
            results[str(pct)] = {"detected": detected, "total_insiders": total_insiders, "rate": rate}
            print(f"{pct}th pct:  {detected:>10}  {total_insiders:>16}  {rate:>16.4f}")

        bar_labels = [f"{p}th" for p in percentiles]
        rates = [results[str(p)]["rate"] for p in percentiles]
        detected_counts = [results[str(p)]["detected"] for p in percentiles]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(bar_labels, rates, color="steelblue", alpha=0.85)
        for bar, det in zip(bars, detected_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                     f"{det}/{total_insiders}", ha="center", va="bottom")
        plt.xlabel("Percentile Threshold")
        plt.ylabel("User-Level Detection Rate")
        plt.title("Insider User Detection Rate at Operational Thresholds")
        plt.ylim(0, 1.15)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "user_detection_rate.png"), dpi=150)
        plt.show()

        return results


    def compute_alert_volume(self, latent_embeddings: np.ndarray, insider_labels: pd.Series | np.ndarray, test_df: pd.DataFrame, train_score_dist: np.ndarray, percentiles: list = [80, 90, 95], save_path: str | None = None) -> dict[str, dict]:
        """
        Computes daily alert volume and alert precision at each operational threshold. Renders a
        grouped bar chart of alerts per day and alert precision per threshold.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim)
            insider_labels: Binary labels (1 = insider, 0 = normal)
            test_df: DataFrame used for date range to derive the number of test days
            train_score_dist: Anomaly scores from the training distribution for threshold derivation
            percentiles: List of percentile thresholds
            save_path: Optional path to persist the figure as PNG

        Returns:
            dict: {percentile_str: {"total_flagged": int, "alerts_per_day": float,
                   "true_positives": int, "alert_precision": float}}
        """
        scores = self.anomaly_score(latent_embeddings)

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        thresholds = {p: float(np.percentile(train_score_dist, p)) for p in percentiles}
        day_col = pd.to_datetime(test_df["day"])
        n_test_days = (day_col.max() - day_col.min()).days + 1

        results = {}
        print(f"\nTest period: {n_test_days} days")
        print(f"\n{'Percentile':>12}  {'Flagged':>10}  {'Alerts/Day':>12}  {'TPs':>8}  {'Precision':>12}")
        print("-" * 60)

        for pct, thresh in thresholds.items():
            flagged = scores >= thresh
            total_flagged = int(flagged.sum())
            tps = int(insider_labels[flagged].sum())
            alerts_per_day = total_flagged / n_test_days
            alert_precision = tps / total_flagged if total_flagged > 0 else 0.0
            results[str(pct)] = {
                "total_flagged": total_flagged,
                "alerts_per_day": alerts_per_day,
                "true_positives": tps,
                "alert_precision": alert_precision,
            }
            print(f"{pct}th pct:  {total_flagged:>10}  {alerts_per_day:>12.2f}  {tps:>8}  {alert_precision:>12.4f}")

        bar_labels = [f"{p}th" for p in percentiles]
        alerts_per_day_vals = [results[str(p)]["alerts_per_day"] for p in percentiles]
        precisions = [results[str(p)]["alert_precision"] for p in percentiles]
        x_pos = np.arange(len(bar_labels))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.bar(x_pos - width / 2, alerts_per_day_vals, width, label="Alerts/Day", color="#3a86a8", alpha=0.85)
        ax2.bar(x_pos + width / 2, precisions, width, label="Alert Precision", color="#e84545", alpha=0.85)
        ax1.set_xlabel("Percentile Threshold")
        ax1.set_ylabel("Alerts per Day", color="#3a86a8")
        ax2.set_ylabel("Alert Precision", color="#e84545")
        ax2.set_ylim(0, max(precisions) * 2 + 0.01 if max(precisions) > 0 else 0.1)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bar_labels)
        ax1.tick_params(axis="y", labelcolor="#3a86a8")
        ax2.tick_params(axis="y", labelcolor="#e84545")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
        plt.title("Alert Volume and Precision at Operational Thresholds")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "alert_volume.png"), dpi=150)
        plt.show()

        return results


    def compute_time_to_first_alert(self, latent_embeddings: np.ndarray, test_df: pd.DataFrame, insiders_df: pd.DataFrame, threshold: float, save_path: str | None = None) -> dict[str, dict]:
        """
        For each insider whose threat window overlaps the test period, finds the first day their
        anomaly score exceeds the threshold. Reports detection lag relative to threat window start.
        Renders a horizontal bar chart of days to detection per insider user.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim); rows aligned with test_df
            test_df: DataFrame with columns 'user' and 'day'; rows aligned with latent_embeddings
            insiders_df: Insider metadata with columns 'user', 'start_day', 'end_day'
            threshold: Anomaly score threshold for flagging (e.g. 90th percentile of training distribution)
            save_path: Optional path to persist the figure as PNG

        Returns:
            dict: {user: {"first_alert_day": date | None, "threat_start": date,
                   "days_delta": int | None, "in_window": bool, "peak_score": float}}
        """
        scores = self.anomaly_score(latent_embeddings)

        scored_df = test_df[["user", "day"]].copy()
        scored_df["score"] = scores
        scored_df["day"] = pd.to_datetime(scored_df["day"])
        scored_df["user"] = scored_df["user"].str.strip().str.lower()

        test_start = scored_df["day"].min()
        test_end = scored_df["day"].max()

        insiders_df = insiders_df.copy()
        insiders_df["start_day"] = pd.to_datetime(insiders_df["start_day"])
        insiders_df["end_day"] = pd.to_datetime(insiders_df["end_day"])
        insiders_df["user_key"] = insiders_df["user"].str.strip().str.lower()

        active_mask = (insiders_df["start_day"] <= test_end) & (insiders_df["end_day"] >= test_start)
        active_insiders = insiders_df.loc[active_mask].copy()

        results = {}
        print(f"\nThreshold: {threshold:.4f}")
        print(f"\n{'User':>15}  {'Threat Start':>14}  {'First Alert':>14}  {'Days Delta':>12}  {'In Window':>10}  {'Peak Score':>12}")
        print("-" * 85)

        for _, row in active_insiders.iterrows():
            user = row["user_key"]
            threat_start = row["start_day"]
            threat_end = row["end_day"]
            user_df = scored_df[scored_df["user"] == user].sort_values("day")
            peak_score = float(user_df["score"].max()) if not user_df.empty else float("nan")
            flagged = user_df[user_df["score"] >= threshold]
            if not flagged.empty:
                first_alert_day = flagged["day"].min()
                days_delta = (first_alert_day - threat_start).days
                in_window = bool(threat_start <= first_alert_day <= threat_end)
            else:
                first_alert_day = None
                days_delta = None
                in_window = False
            results[user] = {
                "first_alert_day": first_alert_day,
                "threat_start": threat_start,
                "days_delta": days_delta,
                "in_window": in_window,
                "peak_score": peak_score,
            }
            alert_str = first_alert_day.strftime("%Y-%m-%d") if first_alert_day is not None else "Never"
            delta_str = str(days_delta) if days_delta is not None else "—"
            print(f"{user:>15}  {threat_start.strftime('%Y-%m-%d'):>14}  {alert_str:>14}  {delta_str:>12}  {str(in_window):>10}  {peak_score:>12.4f}")

        users_list = list(results.keys())
        bar_vals = []
        colors_list = []
        for u in users_list:
            d = results[u]["days_delta"]
            if d is None:
                colors_list.append("#aaaaaa")
                bar_vals.append(1)
            elif results[u]["in_window"]:
                colors_list.append("#e84545")
                bar_vals.append(max(d, 1))
            else:
                colors_list.append("#d4a017")
                bar_vals.append(max(d, 1))

        fig_height = max(4, len(users_list) * 0.6 + 1)
        plt.figure(figsize=(10, fig_height))
        y_pos = np.arange(len(users_list))
        plt.barh(y_pos, bar_vals, color=colors_list, alpha=0.85)
        plt.yticks(y_pos, users_list)
        for i, u in enumerate(users_list):
            if results[u]["days_delta"] is None:
                plt.text(bar_vals[i] + 0.1, i, "Never detected", va="center", fontsize=8, color="#888888")
        plt.xlabel("Days from Threat Window Start to First Alert")
        plt.title(f"Time to First Alert per Insider (threshold = {threshold:.4f})")
        plt.axvline(0, color="black", linewidth=0.8, alpha=0.5)
        plt.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "time_to_first_alert.png"), dpi=150)
        plt.show()

        return results


    def compute_rank_order(self, latent_embeddings: np.ndarray, test_df: pd.DataFrame, insider_users: set | list, top_n: int = 10, save_path: str | None = None) -> pd.DataFrame:
        """
        Ranks all users in the test stream by peak anomaly score and reports where insider users
        appear in the ranking. Renders a scatter plot of rank vs peak score with insiders highlighted.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim); rows aligned with test_df
            test_df: DataFrame with 'user' column; rows aligned with latent_embeddings
            insider_users: Collection of insider user identifiers (lowercase, stripped)
            top_n: Number of top-ranked users to include in the printed summary table
            save_path: Optional path to persist the figure as PNG

        Returns:
            pd.DataFrame: All users sorted by peak score descending with columns
                          ['user', 'peak_score', 'rank', 'is_insider']
        """
        scores = self.anomaly_score(latent_embeddings)

        scored_df = test_df[["user"]].copy()
        scored_df["score"] = scores
        scored_df["user"] = scored_df["user"].str.strip().str.lower()

        insider_set = {u.strip().lower() for u in insider_users}

        user_peak = scored_df.groupby("user")["score"].max().reset_index()
        user_peak.columns = ["user", "peak_score"]
        user_peak = user_peak.sort_values("peak_score", ascending=False).reset_index(drop=True)
        user_peak["rank"] = user_peak.index + 1
        user_peak["is_insider"] = user_peak["user"].isin(insider_set)

        total_users = len(user_peak)
        insider_rows = user_peak[user_peak["is_insider"]]

        print(f"\nTotal users in test stream: {total_users}")
        print("\nInsider users in ranking:")
        print(f"\n{'Rank':>8}  {'User':>15}  {'Peak Score':>12}")
        print("-" * 40)
        for _, row in insider_rows.iterrows():
            print(f"{int(row['rank']):>8}  {row['user']:>15}  {row['peak_score']:>12.4f}")

        print(f"\nTop-{top_n} users by peak anomaly score:")
        print(f"\n{'Rank':>8}  {'User':>15}  {'Peak Score':>12}  {'Insider':>10}")
        print("-" * 50)
        for _, row in user_peak.head(top_n).iterrows():
            print(f"{int(row['rank']):>8}  {row['user']:>15}  {row['peak_score']:>12.4f}  {str(row['is_insider']):>10}")

        normal_mask = ~user_peak["is_insider"]
        plt.figure(figsize=(10, 5))
        plt.scatter(user_peak.loc[normal_mask, "rank"], user_peak.loc[normal_mask, "peak_score"],
                    color="#3a86a8", alpha=0.4, s=8, label="Normal")
        if not insider_rows.empty:
            plt.scatter(insider_rows["rank"], insider_rows["peak_score"],
                        color="#e84545", s=60, zorder=5, label="Insider")
            for _, row in insider_rows.iterrows():
                plt.annotate(row["user"], (row["rank"], row["peak_score"]),
                             textcoords="offset points", xytext=(5, 3), fontsize=7, color="#e84545")
        plt.xlabel("Rank (1 = Highest Anomaly Score)")
        plt.ylabel("Peak Anomaly Score")
        plt.title(f"User Rank-Order by Peak Anomaly Score ({total_users} users)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "rank_order.png"), dpi=150)
        plt.show()

        return user_peak


    def compute_score_distribution_shift(self, latent_embeddings: np.ndarray, reference_score_dist: np.ndarray, thresholds: dict | None = None, save_path: str | None = None) -> tuple[float, float]:
        """
        Runs a Kolmogorov-Smirnov test comparing a reference score distribution against the
        scores produced from the supplied embeddings. Renders overlaid histograms with optional
        percentile threshold lines.

        Args:
            latent_embeddings: Latent embedding matrix of shape (n_samples, latent_emb_dim)
            reference_score_dist: Reference anomaly score distribution. Pass the clean calibration
                baseline when verifying that operational thresholds will produce the expected
                alert rates on evaluation data.
            thresholds: Optional dict {percentile: score_value} to overlay as vertical lines
            save_path: Optional path to persist the figure as 'score_distribution_shift.png'

        Returns:
            tuple[float, float]: (ks_stat, ks_pval)
        """
        eval_scores = self.anomaly_score(latent_embeddings)
        ks_stat, ks_pval = ks_2samp(reference_score_dist, eval_scores)

        print("\nKolmogorov-Smirnov Distribution Shift Test")
        print(f"  KS Statistic : {ks_stat:.4f}")
        print(f"  p-value      : {ks_pval:.6f}")
        if ks_pval < 0.05:
            print("  WARNING: Significant distribution shift detected (p < 0.05)")
        else:
            print("  Distributions are consistent (p >= 0.05)")

        plt.figure(figsize=(10, 5))
        plt.hist(reference_score_dist, bins=60, color="#3a86a8", alpha=0.5, density=True,
                 label="Reference (clean baseline)")
        plt.hist(eval_scores, bins=60, color="#e84545", alpha=0.5, density=True,
                 label="Evaluation")
        if thresholds:
            threshold_colors = ["#28a745", "#d4a017", "#ff1744"]
            for (pct, thresh), col in zip(sorted(thresholds.items()), threshold_colors):
                plt.axvline(thresh, color=col, linestyle="--", alpha=0.8,
                            label=f"{pct}th pct ({thresh:.4f})")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Density")
        plt.title(f"Score Distribution: Reference vs Evaluation (KS={ks_stat:.4f}, p={ks_pval:.4f})")
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "score_distribution_shift.png"), dpi=150)
        plt.show()

        return ks_stat, ks_pval
