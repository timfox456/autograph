import pandas as pd
import numpy as np
import joblib
import os
from .base import IdentityModel, AnomalyModel
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from src.utils.pipeline import create_classification_pipeline

try:
    import xgboost as xgb
except ImportError:
    xgb = None

class XGBoostMatcher(IdentityModel):
    def __init__(self, n_estimators=100, max_depth=15, 
                 learning_rate=0.3, subsample=0.9,
                 colsample_bytree=0.6, gamma=0,
                 min_child_weight=3, random_state=42, **kwargs):
        """
        Initialize XGBoostMatcher with tuned hyperparameters.
        
        Default parameters from hyperparameter tuning (n_iter=60, F1: 0.9884):
        - n_estimators: 100
        - max_depth: 15 (was 6)
        - learning_rate: 0.3 (was 0.1)
        - subsample: 0.9 (was 1.0)
        - colsample_bytree: 0.6 (was 1.0)
        - gamma: 0
        - min_child_weight: 3 (was 1)
        """
        super().__init__()
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': random_state,
            **{k: v for k, v in kwargs.items() if k != 'num_class'}
        }
        
        xgb_model = XGBClassifier(**self.params)
        self.pipeline = create_classification_pipeline(xgb_model, use_scaler=True)
        
        self.label_encoder = LabelEncoder()
        self.raw_features = []
        self.features = None
        self._trained = False

    def train(self, X, y, feature_names: list[str]):
        if xgb is None:
            raise ImportError("xgboost is not installed.")
        
        self.raw_features = feature_names
        self.features = self._sanitize_feature_names(feature_names)
        
        self.label_encoder.fit(y)
        encoded_y = self.label_encoder.transform(y)
        
        num_class = len(self.label_encoder.classes_)
        self.pipeline.named_steps['classifier'].set_params(num_class=num_class)
        
        X_df = self._prepare_features(X)
        
        self.pipeline.fit(X_df, encoded_y)
        self._trained = True

    def _prepare_features(self, X_df):
        """
        Overrides base to handle sanitization.
        """
        X_prepared = super()._prepare_features(X_df, expected_features=self.raw_features)
        
        X_prepared.columns = self.features
        
        X_prepared = X_prepared.apply(pd.to_numeric, errors='coerce').fillna(0)
        return X_prepared

    def predict_probs_batch(self, X_df) -> list[list[tuple[str, float]]]:
        if self.pipeline is None:
            raise ValueError("Model not trained.")
        
        X_test = self._prepare_features(X_df)
        
        probs = self.pipeline.predict_proba(X_test)
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

    def evaluate_cv(self, X, y, cv=5):
        """
        Evaluate model using StratifiedKFold cross-validation.
        Returns train and validation scores for multiple metrics.
        """
        from sklearn.model_selection import StratifiedKFold, cross_validate
        
        encoded_y = self.label_encoder.fit_transform(y)
        
        if not self.features:
            raw_feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
            self.raw_features = raw_feature_names
            self.features = self._sanitize_feature_names(raw_feature_names)
        
        X_prepared = self._prepare_features(X)
        
        num_class = len(self.label_encoder.classes_)
        self.pipeline.named_steps['classifier'].set_params(num_class=num_class)
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_validate(
            self.pipeline, X_prepared, encoded_y, cv=skf,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            return_train_score=True
        )
        return scores

    
    def save(self, path):
        if d := os.path.dirname(path):
            os.makedirs(d, exist_ok=True)
        joblib.dump({
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'features': self.features,
            'raw_features': self.raw_features,
            'params': self.params
        }, path)

    def load(self, path):
        data = joblib.load(path)
        if 'pipeline' in data:
            self.pipeline = data['pipeline']
        elif 'model' in data:
            raise ValueError(
                "Legacy XGBoost model format detected (xgb.Booster). "
                "This format is incompatible with the current Pipeline-based architecture. "
                "Please retrain the model using train_models.py."
            )
        else:
            raise KeyError("Neither 'pipeline' nor 'model' key found in saved model data")
        self.label_encoder = data['label_encoder']
        self.features = data['features']
        self.raw_features = data.get('raw_features', self.features)
        self.params = data['params']
        self._trained = True

    def is_trained(self) -> bool:
        return bool(self._trained and self.pipeline is not None and self.label_encoder is not None and self.features)

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
        self.features = self._sanitize_feature_names(feature_names)

        unique_identities = identities.unique()
        X_df_all = self._prepare_features(X)

        for identity in unique_identities:
            labels = (identities == identity).astype(int)
            
            dtrain = xgb.DMatrix(X_df_all.values, label=labels, feature_names=self.features)
            self.models[identity] = xgb.train(self.params, dtrain)
        self._trained = True

    def _prepare_features(self, X_df):
        """
        Overrides base to handle sanitization.
        """
        X_prepared = super()._prepare_features(X_df, expected_features=self.raw_features)
        
        X_prepared.columns = self.features
        
        X_prepared = X_prepared.apply(pd.to_numeric, errors='coerce').fillna(0)
        return X_prepared

    def score(self, identity: str, X_df: pd.DataFrame) -> tuple[object, object]:
        if identity not in self.models:
            return None, None
        
        model = self.models[identity]
        X_test = self._prepare_features(X_df)
        
        dtest = xgb.DMatrix(X_test.values, feature_names=self.features)
        probs = model.predict(dtest)
        
        score = float(probs[0])
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
