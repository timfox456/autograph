import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class IdentityMatcher:
    def __init__(self, model_path=None):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.model_path = model_path
        self.features = []

    def train(self, data_path):
        df = pd.read_csv(data_path)
        
        # Identity is our target
        y = self.label_encoder.fit_transform(df['identity'])
        
        # Features are everything except label, identity, filename
        X = df.drop(columns=['label', 'identity', 'filename'])
        self.features = X.columns.tolist()
        
        self.model.fit(X, y)
        print(f"Model trained on {len(X)} samples with {len(self.features)} features.")
        
        if self.model_path:
            self.save(self.model_path)

    def predict(self, feature_dict):
        """
        Predicts identity from a dictionary of features.
        Handles missing features by filling with 0 (consistent with training data processing).
        """
        # Create a single row DataFrame with same columns as training
        X_test = pd.DataFrame([feature_dict])

        # Identify missing columns and add them all at once to prevent DataFrame fragmentation
        missing_cols = [col for col in self.features if col not in X_test.columns]
        if missing_cols:
            missing_data = pd.DataFrame([[0] * len(missing_cols)], columns=missing_cols)
            X_test = pd.concat([X_test, missing_data], axis=1)

        # Reorder columns to match training
        X_test = X_test[self.features]
        
        probs = self.model.predict_proba(X_test)[0]
        classes = self.label_encoder.classes_
        
        results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        return results

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
