import pandas as pd
import joblib
import json
import os
from .base import IdentityModel
from .supervised import RandomForestMatcher
from .xgboost_models import XGBoostMatcher
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder


class EnsembleMatcher(IdentityModel):
    """
    Ensemble matcher combining RandomForestMatcher and XGBoostMatcher using soft voting.
    Uses tuned hyperparameters from JSON configuration files.
    """

    def __init__(self, rf_params_path=None, xgb_params_path=None, random_state=42):
        """
        Initialize EnsembleMatcher with tuned parameters from JSON files.

        Args:
            rf_params_path: Path to best_params_rf.json
            xgb_params_path: Path to best_params_xgb.json
            random_state: Random state for reproducibility
        """
        super().__init__()
        self.rf_params_path = rf_params_path
        self.xgb_params_path = xgb_params_path
        self.random_state = random_state

        self.rf_params = self._load_params(rf_params_path) if rf_params_path else {}
        self.xgb_params = self._load_params(xgb_params_path) if xgb_params_path else {}

        self.rf_matcher = RandomForestMatcher(
            **self._extract_classifier_params(self.rf_params), random_state=random_state
        )
        self.xgb_matcher = XGBoostMatcher(
            **self._extract_classifier_params(self.xgb_params), random_state=random_state
        )

        self.voting_classifier = None
        self.label_encoder = LabelEncoder()
        self.features = None
        self._trained = False

    def _load_params(self, path):
        """Load parameters from JSON file."""
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("best_params", {})

    def _extract_classifier_params(self, params_dict):
        """
        Extract classifier parameters from pipeline params dict.
        Removes 'classifier__' prefix.

        Note: The model_type parameter was previously accepted but unused since
        both RF and XGB use the same parameter naming convention (classifier__ prefix).
        This has been removed to simplify the API.
        """
        extracted = {}
        prefix = "classifier__"
        for key, value in params_dict.items():
            if key.startswith(prefix):
                param_name = key[len(prefix) :]
                extracted[param_name] = value
        return extracted

    def train(self, X, y, feature_names: list[str]):
        """
        Train the ensemble on features X and labels y.

        NOTE: Label encoding strategy - Each sub-model (RF, XGB) internally fits
        its own LabelEncoder during train(). The VotingClassifier receives the
        ensemble's encoded labels. This is intentional: each model manages its
        own encoding, and all produce consistent predictions because they decode
        using their fitted encoders. This design allows sub-models to be used
        independently or within the ensemble without requiring pre-encoded labels.

        Args:
            X_data: Feature matrix (DataFrame or array-like)
            y: Target labels (raw string labels, will be encoded internally)
            feature_names: List of feature names
        """
        self.features = feature_names
        self.label_encoder.fit(y)
        encoded_y = self.label_encoder.transform(y)

        X_prepared = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)

        # Train sub-models with RAW labels (they handle internal encoding)
        # This is intentional - each sub-model fits its own LabelEncoder
        self.rf_matcher.train(X_prepared, y, feature_names)
        self.xgb_matcher.train(X_prepared, y, feature_names)

        # VotingClassifier uses the ensemble's encoded labels
        # All models will produce consistent predictions because each decodes
        # using their own fitted encoder
        self.voting_classifier = VotingClassifier(
            estimators=[("rf", self.rf_matcher.pipeline), ("xgb", self.xgb_matcher.pipeline)],
            voting="soft",
        )

        self.voting_classifier.fit(X_prepared, encoded_y)
        self._trained = True

    def predict_probs_batch(self, X_df) -> list[list[tuple[str, float]]]:
        """
        Predict identity probabilities for multiple samples using soft voting.
        """
        if not self._trained or self.voting_classifier is None:
            raise ValueError("Model not trained.")

        X_test = self._prepare_features(X_df)

        probs = self.voting_classifier.predict_proba(X_test)
        classes = self.label_encoder.classes_

        all_results = []
        for sample_probs in probs:
            results = sorted(zip(classes, sample_probs), key=lambda x: x[1], reverse=True)
            all_results.append(results)
        return all_results

    def predict_probs(self, X_df) -> list[tuple[str, float]]:
        """Predict identity probabilities for a single sample."""
        return self.predict_probs_batch(X_df)[0]

    def save(self, path):
        """Save the ensemble model to disk."""
        if d := os.path.dirname(path):
            os.makedirs(d, exist_ok=True)

        joblib.dump(
            {
                "voting_classifier": self.voting_classifier,
                "rf_matcher": self.rf_matcher,
                "xgb_matcher": self.xgb_matcher,
                "label_encoder": self.label_encoder,
                "features": self.features,
                "rf_params": self.rf_params,
                "xgb_params": self.xgb_params,
            },
            path,
        )

    def load(self, path):
        """Load the ensemble model from disk."""
        data = joblib.load(path)
        self.voting_classifier = data["voting_classifier"]
        self.rf_matcher = data["rf_matcher"]
        self.xgb_matcher = data["xgb_matcher"]
        self.label_encoder = data["label_encoder"]
        self.features = data["features"]
        self.rf_params = data.get("rf_params", {})
        self.xgb_params = data.get("xgb_params", {})
        self._trained = True

    def is_trained(self) -> bool:
        """Check if the model has been trained/loaded."""
        return bool(self._trained and self.voting_classifier is not None and self.features)
