import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

class UEBAIsolationForest:
    """
    Isolation Forest for anomaly detection on Autoencoder latent embeddings.
    """
    
    def __init__(self, n_estimators: int=200, max_samples: str="auto", contamination: float=0.05, random_state: int=42) -> None:
        """
        Initializes the Isolation Forest.
        
        Args:
            n_estimators: The number of trees in the forest
            max_samples: The subsamples size for each tree
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
        Trains the Isolation Forest on latent embeddibgs.
        
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