import pytest
import pandas as pd
import numpy as np
from src.models.xgboost_models import XGBoostMatcher
import os

def test_xgboost_matcher_basic_flow(tmp_path):
    """Test training and prediction with XGBoostMatcher."""
    # Create synthetic data
    X = pd.DataFrame({
        'feature1': [1, 2, 1, 2, 5, 6, 5, 6],
        'feature2': [0.1, 0.2, 0.1, 0.2, 0.8, 0.9, 0.8, 0.9]
    })
    y = pd.Series(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    feature_names = ['feature1', 'feature2']

    matcher = XGBoostMatcher()
    matcher.train(X, y, feature_names)

    # Test prediction
    test_df = pd.DataFrame({'feature1': [1.5], 'feature2': [0.15]})
    probs = matcher.predict_probs(test_df)

    assert len(probs) == 2  # Two classes A and B
    assert probs[0][0] == 'A'
    assert probs[0][1] > 0.5

def test_xgboost_matcher_feature_cleaning():
    """Test that XGBoostMatcher handles illegal characters in feature names."""
    X = pd.DataFrame({
        'node[type]': [1, 2],
        'count<5>': [0.1, 0.2]
    })
    y = pd.Series(['A', 'B'])
    feature_names = ['node[type]', 'count<5>']

    matcher = XGBoostMatcher()
    # This should not raise "feature_names may not contain [, ] or <"
    matcher.train(X, y, feature_names)
    
    # Check that features were cleaned
    assert 'node_type_' in matcher.features
    assert 'count_5_' in matcher.features

    # Prediction should also work with original names (it cleans them internally)
    test_df = pd.DataFrame({'node[type]': [1], 'count<5>': [0.1]})
    probs = matcher.predict_probs(test_df)
    assert len(probs) == 2

def test_xgboost_matcher_missing_features():
    """Test that XGBoostMatcher handles missing features during prediction."""
    X = pd.DataFrame({
        'f1': [1, 2],
        'f2': [3, 4]
    })
    y = pd.Series(['A', 'B'])
    feature_names = ['f1', 'f2']

    matcher = XGBoostMatcher()
    matcher.train(X, y, feature_names)

    # Predict with missing f2
    test_df = pd.DataFrame({'f1': [1]})
    probs = matcher.predict_probs(test_df)
    
    assert len(probs) == 2
    assert probs[0][1] > 0 # Should still return probabilities

def test_xgboost_matcher_save_load(tmp_path):
    """Test saving and loading XGBoostMatcher."""
    X = pd.DataFrame({'f1': [1, 5], 'f2': [2, 6]})
    y = pd.Series(['A', 'B'])
    feature_names = ['f1', 'f2']

    matcher = XGBoostMatcher()
    matcher.train(X, y, feature_names)
    
    model_path = tmp_path / "xgb.joblib"
    matcher.save(str(model_path))

    # Load into new instance
    new_matcher = XGBoostMatcher()
    new_matcher.load(str(model_path))

    assert new_matcher.features == matcher.features
    assert list(new_matcher.label_encoder.classes_) == ['A', 'B']

    # Predict with loaded model
    test_df = pd.DataFrame({'f1': [1], 'f2': [2]})
    probs = new_matcher.predict_probs(test_df)
    assert probs[0][0] == 'A'

def test_xgboost_matcher_numeric_conversion():
    """Test that XGBoostMatcher handles non-numeric data gracefully by coercing."""
    X = pd.DataFrame({
        'f1': [1, 2, 'invalid'],
        'f2': [0.1, 0.2, 0.3]
    })
    y = pd.Series(['A', 'A', 'B'])
    feature_names = ['f1', 'f2']

    matcher = XGBoostMatcher()
    # Should not crash on 'invalid' string
    matcher.train(X, y, feature_names)
    
    test_df = pd.DataFrame({'f1': ['foo'], 'f2': [0.1]})
    probs = matcher.predict_probs(test_df)
    assert len(probs) == 2
