import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from .base import IdentityModel
from src.utils.pipeline import create_classification_pipeline


class RandomForestMatcher(IdentityModel):
    def __init__(
        self,
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=False,
        random_state=42,
        model_path=None,
    ):
        """
        Initialize RandomForestMatcher with tuned hyperparameters.

        Default parameters from hyperparameter tuning (n_iter=60, F1: 0.7059):
        - n_estimators: 500 (was 100)
        - max_depth: None (unlimited)
        - min_samples_split: 5 (was 2)
        - min_samples_leaf: 2 (was 1)
        - max_features: 'sqrt' (was default)
        - bootstrap: False (was True)
        """
        super().__init__()
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight="balanced",
            random_state=random_state,
        )
        self.pipeline = create_classification_pipeline(classifier, use_scaler=True)
        self.label_encoder = LabelEncoder()
        self.features = None
        self.model_path = model_path
        self._trained = False

    def train(self, X, y, feature_names: list[str]):
        self.features = feature_names
        self.label_encoder.fit(y)
        encoded_y = self.label_encoder.transform(y)
        # Pipeline handles scaling internally - scaler is fit on training data only
        self.pipeline.fit(X, encoded_y)
        self._trained = True

    def predict_probs_batch(self, X_df) -> list[list[tuple[str, float]]]:
        """
        Predicts identity probabilities for multiple samples.
        """
        X_test = self._prepare_features(X_df)
        probs = self.pipeline.predict_proba(X_test)
        classes = self.label_encoder.classes_

        all_results = []
        for sample_probs in probs:
            results = sorted(zip(classes, sample_probs), key=lambda x: x[1], reverse=True)
            all_results.append(results)
        return all_results

    def predict_probs(self, X_df) -> list[tuple[str, float]]:
        return self.predict_probs_batch(X_df)[0]

    def evaluate_cv(self, X, y, cv=5):
        """
        Evaluate model using StratifiedKFold cross-validation.
        Returns train and validation scores for multiple metrics.

        NOTE: This method uses a local LabelEncoder to avoid mutating the
        model's fitted state. The model's label_encoder is NOT re-fitted here.
        """
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.preprocessing import LabelEncoder

        # Use a local label encoder to avoid mutating model state
        local_encoder = LabelEncoder()
        encoded_y = local_encoder.fit_transform(y)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_validate(
            self.pipeline,
            X,
            encoded_y,
            cv=skf,
            scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            return_train_score=True,
        )
        return scores

    def save(self, path):
        if d := os.path.dirname(path):
            os.makedirs(d, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "label_encoder": self.label_encoder,
                "features": self.features,
            },
            path,
        )

    def load(self, path):
        data = joblib.load(path)
        # Handle both old 'model' key and new 'pipeline' key for backward compatibility
        if "pipeline" in data:
            self.pipeline = data["pipeline"]
        elif "model" in data:
            import warnings

            warnings.warn(
                "Loading legacy RandomForest model (pre-pipeline format). "
                "Scaling will not be applied to predictions. "
                "Consider retraining with the current pipeline for best results.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Wrap the raw classifier in a no-scaler pipeline for API compatibility
            from src.utils.pipeline import create_classification_pipeline

            self.pipeline = create_classification_pipeline(data["model"], use_scaler=False)
        else:
            raise KeyError("Neither 'pipeline' nor 'model' key found in saved data")
        self.label_encoder = data["label_encoder"]
        self.features = data["features"]
        self._trained = True

    def is_trained(self) -> bool:
        return bool(self._trained and self.features)


class IdentityMatcher:
    """
    Legacy wrapper for backward compatibility or convenience.
    Defaults to RandomForestMatcher.
    """

    def __init__(self, model_path=None):
        self.implementation = RandomForestMatcher()
        self.model_path = model_path

    def train(self, data_path, oversample: bool = True):
        df = pd.read_csv(data_path)

        if oversample:
            # Oversample minority classes to match the majority class count
            counts = df["identity"].value_counts()
            max_count = counts.max()

            oversampled_dfs = []
            for identity, count in counts.items():
                identity_df = df[df["identity"] == identity]
                if count < max_count:
                    # Duplicate samples to reach max_count
                    identity_df = identity_df.sample(max_count, replace=True, random_state=42)
                oversampled_dfs.append(identity_df)

            df = pd.concat(oversampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Oversampled dataset to {len(df)} samples ({max_count} per identity).")

        y = df["identity"]
        X = df.drop(columns=["label", "identity", "filename"])
        self.implementation.train(X, y, X.columns.tolist())
        self.features = self.implementation.features

        if self.model_path:
            self.save(self.model_path)

    def predict(self, feature_dict):
        X_test = pd.DataFrame([feature_dict])
        return self.implementation.predict_probs(X_test)

    def save(self, path):
        self.implementation.save(path)

    def load(self, path):
        self.implementation.load(path)
        self.features = self.implementation.features
