import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class ConsistencyChecker:
    """
    Maintains one anomaly detection model per identity.
    """
    def __init__(self, models_dir=None):
        self.models = {} # identity -> model
        self.models_dir = models_dir
        self.features = []

    def train(self, data_path):
        df = pd.read_csv(data_path)
        self.features = df.drop(columns=['label', 'identity', 'filename']).columns.tolist()
        
        identities = df['identity'].unique()
        for identity in identities:
            id_data = df[df['identity'] == identity].drop(columns=['label', 'identity', 'filename'])
            
            # IsolationForest needs at least one sample, but ideally more.
            # For the pilot, we'll train on what we have.
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(id_data)
            self.models[identity] = model
            print(f"Consistency model trained for identity: {identity}")
            
        if self.models_dir:
            self.save(self.models_dir)

    def check_consistency(self, identity, feature_dict):
        """
        Returns a score: 1 for consistent, -1 for outlier.
        Also returns a continuous decision_function value (higher is more normal).
        """
        if identity not in self.models:
            return None, None
            
        model = self.models[identity]
        
        # Prepare data
        X_test = pd.DataFrame([feature_dict])
        for col in self.features:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[self.features]
        
        prediction = model.predict(X_test)[0]
        score = model.decision_function(X_test)[0]
        
        return prediction, score

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        joblib.dump({
            'models': self.models,
            'features': self.features
        }, os.path.join(directory, 'consistency_models.joblib'))

    def load(self, directory):
        data = joblib.load(os.path.join(directory, 'consistency_models.joblib'))
        self.models = data['models']
        self.features = data['features']
