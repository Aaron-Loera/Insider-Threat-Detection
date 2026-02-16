import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class Autoencoder:
    """
    Autoencoder for learning latent representation of UEBA behavioral features.
    """
    
    def __init__(self, input_dim: int, latent_dim: int=16, hidden_dim: int=32, learning_rate: float=1e-3) -> None:
        """
        Initializes the autoencoder architecture.
        
        Args:
            input_dim: The number of input features
            latent_dim: The size of latent embeddings
            hidden_dim: The size of hidden layers
            learning_rate: Optimizer learning rate
            
        Returns:
            None:
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
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
        x = layers.Dense(self.hidden_dim, activation="relu")(inputs)
        latent = layers.Dense(self.latent_dim, activation="relu", name="latent_space")(x)
        
        # Decoder construction
        x = layers.Dense(self.hidden_dim, activation="relu")(latent)
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
    
    
    def train(self, x_train: np.ndarray, save_path: str, epochs: int=50, batch_size: int=128, validation_split: float=0.1):
        """
        Trains the autoencoder using the specified hyperparameters.
        
        Args:
            x_train: The scaled UEBA-enhanced feature matrix
            save_path: The path to store the training log
            epochs: The number of epochs to train the autoencoder for
            batch_size: Batch size
            validation_split: The validation data ratio
        """
        # Defining logger to save training history
        log_save_path = os.path.join(save_path, "training_log.csv")
        csv_logger = tf.keras.callbacks.CSVLogger(log_save_path, append=True)
        
        # Training the full autoencoder
        history = self.autoencoder.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=1,
            callbacks=[csv_logger]
        )
        
        return history
        
    
    def encode(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Generates latent embeddings for UEBA data.
        
        Args:
            feature_matrix: The scaled UEBA feature matrix
            
        Returns:
            np.ndarray: The generate latent embeddings
        """
        return self.encoder.predict(feature_matrix)
    
    
    def reconstruction_error(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the reconstruction error per sample.
        
        Args:
            feature_matrix: The scaled UEBA feature matrix
            
        Returns:
            np.ndarray: Reconstruction MSE per sample
        """
        # Reconstructing original feature matrix
        reconstruction = self.autoencoder.predict(feature_matrix)
        
        # Computing the mean sqaured error
        error = np.mean(np.square(feature_matrix - reconstruction), axis=1)
        
        return error