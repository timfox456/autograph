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
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            class_weight='balanced'
        )
        self.label_encoder = LabelEncoder()
        self._trained = False

    def train(self, X, y, feature_names: list[str]):
        self.features = feature_names
        self.label_encoder.fit(y)
        encoded_y = self.label_encoder.transform(y)
        self.model.fit(X, encoded_y)
        self._trained = True

    def predict_probs_batch(self, X_df) -> list[list[tuple[str, float]]]:
        """
        Predicts identity probabilities for multiple samples.
        """
        X_test = self._prepare_features(X_df)
        probs = self.model.predict_proba(X_test)
        classes = self.label_encoder.classes_
        
        all_results = []
        for sample_probs in probs:
            results = sorted(zip(classes, sample_probs), key=lambda x: x[1], reverse=True)
            all_results.append(results)
        return all_results

    def predict_probs(self, X_df) -> list[tuple[str, float]]:
        return self.predict_probs_batch(X_df)[0]

    def save(self, path):
        if d := os.path.dirname(path):
            os.makedirs(d, exist_ok=True)
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
            counts = df['identity'].value_counts()
            max_count = counts.max()
            
            oversampled_dfs = []
            for identity, count in counts.items():
                identity_df = df[df['identity'] == identity]
                if count < max_count:
                    # Duplicate samples to reach max_count
                    identity_df = identity_df.sample(max_count, replace=True, random_state=42)
                oversampled_dfs.append(identity_df)
            
            df = pd.concat(oversampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Oversampled dataset to {len(df)} samples ({max_count} per identity).")

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
