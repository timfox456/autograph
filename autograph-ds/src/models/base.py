from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

class IdentityModel(ABC):
    """
    Interface for identity matching models.
    """
    def __init__(self):
        self.features = []

    @abstractmethod
    def train(self, X, y, feature_names: List[str]):
        """Train the model on features X and labels y."""
        pass

    @abstractmethod
    def predict_probs(self, X_df) -> List[Tuple[str, float]]:
        """Predict identity probabilities for a single sample X (DataFrame)."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save the model to the specified path."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the model from the specified path."""
        pass

class AnomalyModel(ABC):
    """
    Interface for anomaly detection (consistency) models.
    """
    def __init__(self):
        self.features = []

    @abstractmethod
    def train(self, X, feature_names: List[str], identities: Any):
        """Train the model on features X."""
        pass

    @abstractmethod
    def score(self, identity: str, X_df: Any) -> Tuple[Optional[int], Optional[float]]:
        """
        Returns (prediction, score). 
        Prediction: 1 for normal, -1 for anomaly.
        Score: Continuous value where higher is more normal.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save the model to the specified path."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the model from the specified path."""
        pass
