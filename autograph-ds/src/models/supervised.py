import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from .base import IdentityModel

class RandomForestMatcher(IdentityModel):
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.label_encoder = LabelEncoder()

    def train(self, X, y, feature_names: list[str]):
        self.features = feature_names
        self.label_encoder.fit(y)
        encoded_y = self.label_encoder.transform(y)
        self.model.fit(X, encoded_y)

    def predict_probs(self, X_df) -> list[tuple[str, float]]:
        """
        Predicts identity probabilities from a DataFrame of features.
        """
        # Ensure correct column order and handle missing
        X_test = self._prepare_features(X_df)
        
        probs = self.model.predict_proba(X_test)[0]
        classes = self.label_encoder.classes_
        
        results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        return results

    def _prepare_features(self, X_df):
        # Identify missing columns and add them all at once
        missing_cols = [col for col in self.features if col not in X_df.columns]
        if missing_cols:
            missing_data = pd.DataFrame([[0] * len(missing_cols)], columns=missing_cols, index=X_df.index)
            X_test = pd.concat([X_df, missing_data], axis=1)
        else:
            X_test = X_df.copy()

        return X_test[self.features]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'features': self.features
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.features = data['features']

class IdentityMatcher:
    """
    Legacy wrapper for backward compatibility or convenience.
    Defaults to RandomForestMatcher.
    """
    def __init__(self, model_path=None):
        self.implementation = RandomForestMatcher()
        self.model_path = model_path

    def train(self, data_path):
        df = pd.read_csv(data_path)
        y = df['identity']
        X = df.drop(columns=['label', 'identity', 'filename'])
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
