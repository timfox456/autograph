from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils import sanitize_feature_names, validate_input_features


class BaseModel(ABC):
    def __init__(self):
        self.features = []

    def _sanitize_feature_names(self, feature_names: list[str]) -> list[str]:
        """
        Sanitizes feature names for model compatibility.
        
        DEPRECATED: Use sanitize_feature_names from src.utils directly.
        This method is kept for backward compatibility.
        """
        return sanitize_feature_names(feature_names)

    def _validate_input(self, X: pd.DataFrame, context: Optional[str] = None) -> None:
        """
        Validate input features for NaN/inf values.
        
        Args:
            X: Input DataFrame to validate
            context: Optional context string for error messages
            
        Raises:
            ValueError: If validation fails (e.g., all NaN column)
        """
        context_str = f" ({context})" if context else ""
        
        for col in X.columns:
            series = X[col]
            
            # Check for all-NaN columns
            if series.isna().all():
                raise ValueError(
                    f"Column '{col}' contains all NaN values{context_str}. "
                    "Please check your input data."
                )
            
            # Check for infinite values in numeric columns
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                inf_mask = np.isinf(numeric_series)
                inf_count = inf_mask.sum()
                if inf_count > 0:
                    # Log warning but don't fail - inf can be valid in some contexts
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Column '{col}' contains {inf_count} infinite values{context_str}"
                    )
            except (TypeError, ValueError):
                pass  # Non-numeric column, skip inf check

    def _prepare_features(self, X_df: Any, expected_features: Optional[list[str]] = None) -> Any:
        """
        Ensures X_df has all expected features in the correct order.
        Missing features are filled with 0.
        
        Also validates input for NaN/inf issues.
        """
        if not isinstance(X_df, pd.DataFrame):
            # If it's a dict or single sample, convert to DataFrame
            X_df = pd.DataFrame([X_df])

        expected = expected_features if expected_features is not None else self.features
        
        # Validate expected features are provided
        if not expected:
            raise ValueError("No expected features provided. Model may not be trained.")
        
        missing_cols = [col for col in expected if col not in X_df.columns]
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

        # Validate the prepared features
        self._validate_input(X_test[expected], context="feature preparation")
        
        return X_test[expected]


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

    @abstractmethod
    def is_trained(self) -> bool:
        """Returns True if the model has been trained/loaded and is ready."""
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

    @abstractmethod
    def is_trained(self) -> bool:
        """Returns True if the model has been trained/loaded and is ready."""
        pass
