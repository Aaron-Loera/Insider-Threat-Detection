import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

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
            x_val: The scaled and filtered validation feature matrix (i.e., no insider rows). If None,
                   a 10% random validation_split is used as fallback.
        """
        log_save_path = os.path.join(save_path, "training_log.csv")
        csv_logger = CSVLogger(log_save_path, append=False)
        
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        callbacks = [csv_logger, early_stop]
        
        if x_val is not None:
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
        else:
            history = self.autoencoder.fit(
                x_train,
                x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
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


    def compute_recall_thresholds(self, feature_matrix: np.ndarray, insider_labels: pd.Series | np.ndarray, percentiles: list = [80, 90, 95], save_path: str | None = None) -> dict[str, float]:
        """
        Computes recall at several percentile thresholds of reconstruction error. Renders
        a bar chart of recall per percentile with captured/total annotations above each bar.

        Args:
            feature_matrix: The scaled UEBA feature matrix
            insider_labels: Binary labels (1 = insider, 0 = normal)
            percentiles: List of percentile thresholds
            save_path: Optional path to persist the figure as PNG

        Returns:
            dict: Recall values {percentile: recall score}
        """
        errors = self.reconstruction_error(feature_matrix)

        if isinstance(insider_labels, pd.Series):
            insider_labels = insider_labels.astype(int).values

        values = {}
        captured_per_pct = {}
        total_insiders = insider_labels.sum()

        for pct in percentiles:
            threshold = np.percentile(errors, pct)
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