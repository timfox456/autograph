import pandas as pd
import numpy as np
import joblib
import os
from .base import IdentityModel, AnomalyModel

try:
    import xgboost as xgb
except ImportError:
    xgb = None

class XGBoostMatcher(IdentityModel):
    def __init__(self, **params):
        super().__init__()
        defaults = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
        self.params = {**defaults, **params}
        self.model = None
        self.label_encoder = None
        self.raw_features = [] # original names before sanitization
        self._trained = False

    def train(self, X, y, feature_names: list[str]):
        if xgb is None:
            raise ImportError("xgboost is not installed.")
        
        self.raw_features = feature_names
        # XGBoost DMatrix doesn't like [, ] or < in feature names
        self.features = self._sanitize_feature_names(feature_names)
        
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        encoded_y = self.label_encoder.fit_transform(y)
        
        num_class = len(self.label_encoder.classes_)
        self.params['num_class'] = num_class
        
        # Prepare X
        X_df = self._prepare_features(X)
        
        dtrain = xgb.DMatrix(X_df.values, label=encoded_y, feature_names=self.features)
        self.model = xgb.train(self.params, dtrain)
        self._trained = True

    def _prepare_features(self, X_df):
        """
        Overrides base to handle sanitization.
        """
        # 1. Use base logic with raw_features without mutating state
        X_prepared = super()._prepare_features(X_df, expected_features=self.raw_features)
        
        # 2. Rename to sanitized names
        X_prepared.columns = self.features
        
        # 3. Ensure numeric
        X_prepared = X_prepared.apply(pd.to_numeric, errors='coerce').fillna(0)
        return X_prepared

    def predict_probs_batch(self, X_df) -> list[list[tuple[str, float]]]:
        if self.model is None:
            raise ValueError("Model not trained.")
        
        X_test = self._prepare_features(X_df)
        dtest = xgb.DMatrix(X_test.values, feature_names=self.features)
        
        probs = self.model.predict(dtest)
        if len(probs.shape) == 1:
            probs = probs.reshape(1, -1)
            
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
            'features': self.features,
            'raw_features': self.raw_features,
            'params': self.params
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.features = data['features']
        self.raw_features = data.get('raw_features', self.features)
        self.params = data['params']
        self._trained = True

    def is_trained(self) -> bool:
        return bool(self._trained and self.model is not None and self.label_encoder is not None and self.features)

class XGBoostAnomaly(AnomalyModel):
    """
    XGBoost-based anomaly detection using one-vs-rest binary classification.
    Trains one model per identity to distinguish it from others.
    """
    def __init__(self, **params):
        super().__init__()
        defaults = {
            'objective': 'binary:logistic',
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        self.params = {**defaults, **params}
        self.models = {}
        self.raw_features = []
        self._trained = False

    def train(self, X, feature_names: list[str], identities: pd.Series):
        if xgb is None:
            raise ImportError("xgboost is not installed.")
        
        self.raw_features = feature_names
        # Similar name cleaning as Matcher
        self.features = self._sanitize_feature_names(feature_names)

        unique_identities = identities.unique()
        X_df_all = self._prepare_features(X)

        for identity in unique_identities:
            # Binary labels: 1 for this identity, 0 for others
            labels = (identities == identity).astype(int)
            
            dtrain = xgb.DMatrix(X_df_all.values, label=labels, feature_names=self.features)
            self.models[identity] = xgb.train(self.params, dtrain)
        self._trained = True

    def _prepare_features(self, X_df):
        """
        Overrides base to handle sanitization.
        """
        # 1. Use base logic with raw_features without mutating state
        X_prepared = super()._prepare_features(X_df, expected_features=self.raw_features)
        
        # 2. Rename to sanitized names
        X_prepared.columns = self.features
        
        # 3. Ensure numeric
        X_prepared = X_prepared.apply(pd.to_numeric, errors='coerce').fillna(0)
        return X_prepared

    def score(self, identity: str, X_df: pd.DataFrame) -> tuple[object, object]:
        if identity not in self.models:
            # Align with IsolationForestAnomaly: unknown identity -> (None, None)
            return None, None
        
        model = self.models[identity]
        X_test = self._prepare_features(X_df)
        
        dtest = xgb.DMatrix(X_test.values, feature_names=self.features)
        probs = model.predict(dtest)
        
        # Probability of being this identity
        score = float(probs[0])
        # Using 0.5 as threshold for consistency check
        prediction = 1 if score > 0.5 else -1
        
        return prediction, score

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        joblib.dump({
            'models': self.models,
            'features': self.features,
            'raw_features': self.raw_features,
            'params': self.params
        }, os.path.join(directory, 'consistency_models.joblib'))

    def load(self, directory):
        data = joblib.load(os.path.join(directory, 'consistency_models.joblib'))
        self.models = data['models']
        self.features = data['features']
        self.raw_features = data.get('raw_features', self.features)
        self.params = data['params']
        self._trained = True

    def is_trained(self) -> bool:
        return bool(self._trained and self.models and self.features)
