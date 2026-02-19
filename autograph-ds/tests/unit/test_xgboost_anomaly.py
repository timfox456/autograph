import pytest
import pandas as pd
import numpy as np
from src.models.xgboost_models import XGBoostAnomaly
import os

def test_xgboost_anomaly_basic_flow(tmp_path):
    """Test training and scoring with XGBoostAnomaly."""
    # Create synthetic data: identity 'A' has features around 1, 'B' around 10
    X = pd.DataFrame({
        'f1': [1.0, 1.1, 0.9, 1.05, 10.0, 10.1, 9.9, 10.05],
        'f2': [1.0, 0.9, 1.1, 1.0, 10.0, 11.0, 9.0, 10.0]
    })
    identities = pd.Series(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    feature_names = ['f1', 'f2']

    model = XGBoostAnomaly()
    model.train(X, feature_names, identities)

    # Test scoring normal sample for A
    test_df_normal_a = pd.DataFrame({'f1': [1.0], 'f2': [1.0]})
    pred, score = model.score('A', test_df_normal_a)
    assert pred == 1
    assert score > 0.5

    # Test scoring anomalous sample for A (looks like B)
    test_df_anomaly_a = pd.DataFrame({'f1': [10.0], 'f2': [10.0]})
    pred_anom, score_anom = model.score('A', test_df_anomaly_a)
    assert score_anom < 0.5
    assert pred_anom == -1

def test_xgboost_anomaly_feature_cleaning():
    """Test that XGBoostAnomaly handles illegal characters."""
    # Provide multiple samples per identity for a meaningful fit
    X = pd.DataFrame({
        'node[type]': [1, 1.1, 0.9, 10, 10.1, 9.9],
        'count<5>': [0.1, 0.12, 0.08, 0.9, 0.88, 0.92]
    })
    identities = pd.Series(['A', 'A', 'A', 'B', 'B', 'B'])
    feature_names = ['node[type]', 'count<5>']

    model = XGBoostAnomaly()
    model.train(X, feature_names, identities)
    
    assert 'node_type_' in model.features
    
    # Known-good sample for A should return a valid score
    test_df_a = pd.DataFrame({'node[type]': [1.0], 'count<5>': [0.1]})
    pred_a, score_a = model.score('A', test_df_a)
    assert pred_a in [1, -1]
    assert isinstance(score_a, float)

    # Unknown identity should return (None, None)
    pred_unknown, score_unknown = model.score('UNKNOWN_ID', test_df_a)
    assert pred_unknown is None
    assert score_unknown is None

def test_xgboost_anomaly_save_load(tmp_path):
    X = pd.DataFrame({'f1': [1, 1.1, 0.9, 10], 'f2': [1, 0.9, 1.1, 10]})
    identities = pd.Series(['A', 'A', 'A', 'B'])
    feature_names = ['f1', 'f2']

    model = XGBoostAnomaly()
    model.train(X, feature_names, identities)
    
    # Save to a directory (consistent with engine expectations)
    save_dir = str(tmp_path / "anom_models")
    model.save(save_dir)

    new_model = XGBoostAnomaly()
    new_model.load(save_dir)

    assert new_model.features == model.features
    assert 'A' in new_model.models
    
    test_df = pd.DataFrame({'f1': [1], 'f2': [1]})
    pred, score = new_model.score('A', test_df)
    assert pred == 1
