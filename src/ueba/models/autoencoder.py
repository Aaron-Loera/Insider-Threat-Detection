import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping


class Autoencoder:
    """
    Autoencoder for learning latent representation of UEBA behavioral features.
    """

    def __init__(self, input_dim: int, latent_dim: int=16, hidden_dims: tuple | int=64, learning_rate: float=1e-3) -> None:
        """
        Initializes the autoencoder architecture.

        Args:
            input_dim: The number of input features
            latent_dim: The size of latent embeddings
            hidden_dims: The size of hidden layers. If a tuple is provided each integer is treated as its own hidden layer
            learning_rate: Optimizer learning rate

        Returns:
            None:
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        self.autoencoder, self.encoder = self._build_model()


    def _build_model(self) -> tuple:
        """
        Builds the autoencoder and encoder models. The autoencoder serves as the model to train, whereas the encoder
        will be utilized for extracting behavioral embeddings.

        Args:
            None:

        Returns:
            tuple: A two-element tuple containing the autoencoder and encoder models
        """
        # Encoder construction
        inputs = layers.Input(shape=(self.input_dim,), name="ueba_input")

        if isinstance(self.hidden_dims, int):
            x = layers.Dense(self.hidden_dims, activation="relu")(inputs)
            x = layers.Dropout(0.2)(x)

        elif isinstance(self.hidden_dims, tuple):
            x = layers.Dense(self.hidden_dims[0], activation="relu")(inputs)
            x = layers.Dropout(0.2)(x)
            for i in range(1, len(self.hidden_dims)):
                x = layers.Dense(self.hidden_dims[i], activation="relu")(x)
                x = layers.Dropout(0.2)(x)

        # Latent space
        latent = layers.Dense(self.latent_dim, activation="linear", name="latent_space")(x)

        # Decoder construction
        if isinstance(self.hidden_dims, int):
            x = layers.Dense(self.hidden_dims, activation="relu")(latent)

        elif isinstance(self.hidden_dims, tuple):
            x = layers.Dense(self.hidden_dims[-1], activation="relu")(latent)
            for i in range(len(self.hidden_dims)-2, -1, -1):
                x = layers.Dense(self.hidden_dims[i], activation="relu")(x)

        outputs = layers.Dense(self.input_dim, activation="linear")(x)

        # Defining the autoencoder and encoder
        autoencoder = models.Model(inputs, outputs, name="ueba_autoencoder")
        encoder = models.Model(inputs, latent, name="ueba_encoder")

        # Compiling the autoencoder
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mse"
        )

        return (autoencoder, encoder)


    def train(self, x_train: np.ndarray, save_path: str, epochs: int=100, batch_size: int=256, x_val: np.ndarray=None):
        """
        Trains the autoencoder with early stopping.

        Args:
            x_train: The scaled and filtered training feature matrix (i.e., no insider rows)
            save_path: The path to store the training log and model artifacts
            epochs: Maximum number of epochs
            batch_size: Batch size
            x_val: The scaled and filtered validation feature matrix (i.e., no insider rows).
                Required — a chronological held-out split must be prepared by the caller
                (see prepare_data.prepare_ae_training_data). A random validation_split
                fallback was removed to prevent silent leakage of future data into the baseline.
        """
        if x_val is None:
            raise ValueError(
                "x_val is required. Pass a chronologically held-out, insider-free "
                "validation matrix produced by prepare_ae_training_data; the previous "
                "random validation_split fallback broke temporal integrity."
            )

        log_save_path = os.path.join(save_path, "training_log.csv")
        csv_logger = CSVLogger(log_save_path, append=False)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        callbacks = [csv_logger, early_stop]

        history = self.autoencoder.fit(
            x_train,
            x_train,
            validation_data=(x_val, x_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1,
            callbacks=callbacks
        )

        return history


    def encode(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Generates latent embeddings for UEBA data.

        Args:
            feature_matrix: The scaled UEBA feature matrix

        Returns:
            np.ndarray: The generated latent embeddings
        """
        return self.encoder.predict(feature_matrix)


    def load(self, load_path: str) -> None:
        """
        Loads previously trained autoencoder and encoder models.

        Args:
            load_path: Directory path containing autoencoder_model.keras and encoder_model.keras

        Returns:
            None:
        """
        self.autoencoder = models.load_model(os.path.join(load_path, "autoencoder_model.keras"), compile=False)
        self.encoder = models.load_model(os.path.join(load_path, "encoder_model.keras"), compile=False)


    def reconstruction_error(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the reconstruction error per sample.

        Args:
            feature_matrix: The scaled UEBA feature matrix

        Returns:
            np.ndarray: Reconstruction MSE per sample
        """
        reconstruction = self.autoencoder.predict(feature_matrix)
        error = np.mean(np.square(feature_matrix - reconstruction), axis=1)
        return error


    def compute_roc_auc_score(self, feature_matrix: np.ndarray, insider_labels: pd.Series | np.ndarray, save_path: str | None = None) -> float:
        """
        Computes AUROC of reconstruction errors against insider labels. Renders the ROC
        curve with a random-guess diagonal reference.

        Args:
            feature_matrix: The scaled UEBA feature matrix
            insider_labels: Binary labels (1 = insider, 0 = normal)
            save_path: Optional path to persist the figure as PNG

        Returns:
            float: ROC AUC score
        """
        errors = self.reconstruction_error(feature_matrix)

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        score = roc_auc_score(insider_labels, errors)
        fpr, tpr, _ = roc_curve(insider_labels, errors)

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


    def compute_avg_prec_score(self, feature_matrix: np.ndarray, insider_labels: pd.Series | np.ndarray, save_path: str | None = None) -> float:
        """
        Computes average precision of reconstruction errors against insider labels. Renders
        the Precision-Recall curve with a baseline at the positive-class prevalence.

        Args:
            feature_matrix: The scaled UEBA feature matrix
            insider_labels: Binary labels (1 = insider, 0 = normal)
            save_path: Optional path to persist the figure as PNG

        Returns:
            float: Average precision score
        """
        errors = self.reconstruction_error(feature_matrix)

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        score = average_precision_score(insider_labels, errors)
        precision, recall, _ = precision_recall_curve(insider_labels, errors)
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


    def compute_recall_thresholds(self, feature_matrix: np.ndarray, insider_labels: pd.Series | np.ndarray, percentiles: list = [80, 90, 95], threshold_source: np.ndarray | None = None, save_path: str | None = None) -> dict[str, float]:
        """
        Computes recall at several percentile thresholds of reconstruction error. Renders
        a bar chart of recall per percentile with captured/total annotations above each bar.

        Args:
            feature_matrix: The scaled UEBA feature matrix
            insider_labels: Binary labels (1 = insider, 0 = normal)
            percentiles: List of percentile thresholds
            threshold_source: Optional 1-D error array used to derive percentile thresholds. When
                supplied, thresholds are computed from this distribution (e.g. the insider-free
                calibration baseline) rather than the evaluation errors themselves — avoiding the
                circularity of measuring recall against a threshold defined by the same data.
            save_path: Optional path to persist the figure as PNG

        Returns:
            dict: Recall values {percentile: recall score}
        """
        errors = self.reconstruction_error(feature_matrix)
        source = np.asarray(threshold_source) if threshold_source is not None else errors

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        values = {}
        captured_per_pct = {}
        total_insiders = insider_labels.sum()

        for pct in percentiles:
            threshold = np.percentile(source, pct)
            flagged = errors >= threshold
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
        plt.title("Recall at Percentile Thresholds (Reconstruction Error)")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "recall_percentiles.png"), dpi=150)
        plt.show()

        return values


    def compute_channel_reconstruction_error(
        self,
        feature_matrix: np.ndarray,
        insider_labels: pd.Series | np.ndarray,
        feature_names: list[str],
        save_path: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Computes mean squared reconstruction error per behavioral channel and compares normal
        vs. insider rows. Renders a grouped bar chart with one bar pair per channel.

        Args:
            feature_matrix: The scaled UEBA feature matrix
            insider_labels: Binary labels (1 = insider, 0 = normal)
            feature_names: Ordered list of column names corresponding to each feature_matrix column,
                used to assign each feature dimension to its behavioral channel
            save_path: Optional path to persist the figure as PNG

        Returns:
            dict: Per-channel error summary {channel: {"normal_mse": float, "insider_mse": float, "ratio": float}}
        """
        # Cross-Channel must be checked first: derived flags like `non_primary_pc_usb_flag`
        # would otherwise be greedily attributed to USB (or PC, File, etc.) by the
        # single-channel patterns below.
        _CHANNEL_PATTERNS = [
            ("Cross-Channel", re.compile(r"_flag$")),
            ("Auth",          re.compile(r"logon|logoff")),
            ("File",          re.compile(r"file_|unique_files")),
            ("USB",           re.compile(r"usb_|device_")),
            ("Email",         re.compile(r"email|attachment|recipient")),
            ("HTTP",          re.compile(r"http_|unique_domain")),
            ("PC",            re.compile(r"pcs_used|non_primary_pc")),
        ]

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        reconstruction = self.autoencoder.predict(feature_matrix, verbose=0)
        sq_errors = np.square(feature_matrix - reconstruction)

        normal_idx  = np.where(insider_labels == 0)[0]
        insider_idx = np.where(insider_labels == 1)[0]

        col_to_channel = {}
        for col in feature_names:
            assigned = "Other"
            for name, pat in _CHANNEL_PATTERNS:
                if pat.search(col):
                    assigned = name
                    break
            col_to_channel[col] = assigned

        channel_order = [name for name, _ in _CHANNEL_PATTERNS] + ["Other"]
        results: dict[str, dict[str, float]] = {}
        ch_normal_mse: dict[str, float] = {}
        ch_insider_mse: dict[str, float] = {}

        for ch in channel_order:
            col_indices = [i for i, c in enumerate(feature_names) if col_to_channel[c] == ch]
            if not col_indices:
                continue
            n_mse = float(sq_errors[normal_idx][:, col_indices].mean())
            if insider_idx.size > 0:
                i_mse = float(sq_errors[insider_idx][:, col_indices].mean())
                ratio = i_mse / n_mse if n_mse > 0 else float("nan")
            else:
                i_mse = float("nan")
                ratio = float("nan")
            ch_normal_mse[ch]  = n_mse
            ch_insider_mse[ch] = i_mse
            results[ch] = {"normal_mse": n_mse, "insider_mse": i_mse, "ratio": ratio}

        channels = list(results.keys())
        x_pos    = np.arange(len(channels))
        width    = 0.35

        plt.figure(figsize=(11, 5))
        plt.bar(x_pos - width / 2, [ch_normal_mse[ch] for ch in channels],
                width, label="Normal", color="#3a86a8", alpha=0.85)
        if insider_idx.size > 0:
            plt.bar(x_pos + width / 2, [ch_insider_mse[ch] for ch in channels],
                    width, label="Insider", color="#e84545", alpha=0.85)
        plt.xticks(x_pos, channels, rotation=30, ha="right")
        plt.ylabel("Mean Squared Reconstruction Error")
        plt.title("Per-Channel AE Reconstruction Error: Normal vs Insider")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "channel_reconstruction_error.png"), dpi=150)
        plt.show()

        print(f"\n{'Channel':>15}  {'Normal MSE':>12}  {'Insider MSE':>12}  {'Ratio (I/N)':>12}")
        print("-" * 56)
        for ch in channels:
            r = results[ch]
            print(f"{ch:>15}  {r['normal_mse']:>12.5f}  {r['insider_mse']:>12.5f}  {r['ratio']:>12.2f}")

        return results


def plot_loss(history, save_path) -> None:
    """
    Plots training vs. validation loss with EarlyStopping best-epoch annotation.
    """
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    best_epoch = np.argmin(val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss", marker=".", color="steelblue")
    plt.plot(val_loss, label="Validation Loss", marker=".", color="coral")
    plt.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.xlim(0.0)
    plt.ylim(0.0, 1.0)
    plt.title("Autoencoder Training History (Normal-Only Data)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_history.png"), dpi=150)
    plt.show()
