import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os
from typing import Optional
from .base import AnomalyModel


class IsolationForestAnomaly(AnomalyModel):
    def __init__(self, contamination=0.25, random_state=42):
        """
        Initialize IsolationForest-based anomaly detector.

        Args:
            contamination: Expected proportion of anomalies. Raised from 0.1 to 0.25
                after AI sample expansion (72 → 265 samples) increased identity diversity,
                making per-identity distributions broader and requiring a more permissive
                anomaly threshold to avoid excessive false positives.
            random_state: Random seed for reproducibility.
        """
        super().__init__()
        self.models = {}  # identity -> model
        self.contamination = contamination
        self.random_state = random_state
        self._trained = False

    def train(self, X, feature_names: list[str], identities: pd.Series):
        self.features = feature_names

        unique_identities = identities.unique()
        for identity in unique_identities:
            id_data = X[identities == identity]

            # IsolationForest needs at least one sample
            model = IsolationForest(
                contamination=self.contamination, random_state=self.random_state
            )
            model.fit(id_data)
            self.models[identity] = model
        self._trained = True

    def score(self, identity: str, X_df: pd.DataFrame) -> tuple[Optional[int], Optional[float]]:
        if identity not in self.models:
            return None, None

        model = self.models[identity]
        X_test = self._prepare_features(X_df)

        prediction = int(model.predict(X_test)[0])
        score = float(model.decision_function(X_test)[0])

        return prediction, score

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(
            {"models": self.models, "features": self.features},
            os.path.join(directory, "consistency_models.joblib"),
        )

    def load(self, directory):
        data = joblib.load(os.path.join(directory, "consistency_models.joblib"))
        self.models = data["models"]
        self.features = data["features"]
        self._trained = True

    def is_trained(self) -> bool:
        return bool(self._trained and self.models and self.features)


class ConsistencyChecker:
    """
    Legacy wrapper for backward compatibility.
    """

    def __init__(self, models_dir=None):
        self.implementation = IsolationForestAnomaly()
        self.models_dir = models_dir

    def train(self, data_path):
        df = pd.read_csv(data_path)
        features = df.drop(columns=["label", "identity", "filename"])
        self.implementation.train(features, features.columns.tolist(), df["identity"])

        if self.models_dir:
            self.save(self.models_dir)

    def check_consistency(self, identity, feature_dict):
        X_test = pd.DataFrame([feature_dict])
        return self.implementation.score(identity, X_test)

    def save(self, directory):
        self.implementation.save(directory)

    def load(self, directory):
        self.implementation.load(directory)
