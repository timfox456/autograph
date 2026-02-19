from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

class BaseModel(ABC):
    def __init__(self):
        self.features = []

    def _prepare_features(self, X_df: Any) -> Any:
        """
        Ensures X_df has all expected features in the correct order.
        Missing features are filled with 0.
        """
        import pandas as pd
        import numpy as np

        if not isinstance(X_df, pd.DataFrame):
            # If it's a dict or single sample, convert to DataFrame
            X_df = pd.DataFrame([X_df])

        missing_cols = [col for col in self.features if col not in X_df.columns]
        if missing_cols:
            # Create a DataFrame of zeros for missing columns
            missing_data = pd.DataFrame(
                np.zeros((len(X_df), len(missing_cols))),
                columns=missing_cols,
                index=X_df.index
            )
            X_test = pd.concat([X_df, missing_data], axis=1)
        else:
            X_test = X_df.copy()

        return X_test[self.features]

class IdentityModel(BaseModel):
    """
    Interface for identity matching models.
    """
    @abstractmethod
    def train(self, X, y, feature_names: List[str]):
        """Train the model on features X and labels y."""
        pass

    @abstractmethod
    def predict_probs(self, X_df) -> List[Tuple[str, float]]:
        """Predict identity probabilities for a single sample X (DataFrame)."""
        pass

    def predict_probs_batch(self, X_df) -> List[List[Tuple[str, float]]]:
        """Predict identity probabilities for multiple samples in X_df."""
        # Default implementation: loop over rows
        import pandas as pd
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame([X_df])
        
        results = []
        for i in range(len(X_df)):
            results.append(self.predict_probs(X_df.iloc[[i]]))
        return results

    @abstractmethod
    def save(self, path: str):
        """Save the model to the specified path."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the model from the specified path."""
        pass

class AnomalyModel(BaseModel):
    """
    Interface for anomaly detection (consistency) models.
    """
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
