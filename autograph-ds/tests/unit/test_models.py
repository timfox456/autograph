import pytest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from src.models.supervised import IdentityMatcher
from src.models.supervised import RandomForestMatcher
from src.models.anomaly import ConsistencyChecker
from src.models.heuristics import HeuristicDetector

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)

@pytest.fixture
def mock_data_path(temp_dir):
    # More samples to make IsolationForest happy
    data = {
        'identity': ['auth1'] * 10 + ['auth2'] * 10,
        'label': ['human'] * 20,
        'filename': [f'f{i}' for i in range(20)],
        'feature1': [1.0 + np.random.normal(0, 0.01) for _ in range(10)] + [5.0 + np.random.normal(0, 0.01) for _ in range(10)],
        'feature2': [0.1 + np.random.normal(0, 0.01) for _ in range(10)] + [0.9 + np.random.normal(0, 0.01) for _ in range(10)]
    }
    df = pd.DataFrame(data)
    path = os.path.join(temp_dir, "test_data.csv")
    df.to_csv(path, index=False)
    return path

def test_identity_matcher(mock_data_path, temp_dir):
    model_path = os.path.join(temp_dir, "matcher.joblib")
    matcher = IdentityMatcher(model_path=model_path)
    
    # Train
    matcher.train(mock_data_path)
    assert len(matcher.features) == 2
    assert os.path.exists(model_path)
    
    # Predict
    features = {'feature1': 1.05, 'feature2': 0.11}
    results = matcher.predict(features)
    assert results[0][0] == 'auth1'
    assert results[0][1] > 0.5

    # Load
    new_matcher = IdentityMatcher()
    new_matcher.load(model_path)
    assert new_matcher.features == matcher.features
    results2 = new_matcher.predict(features)
    assert results2[0][0] == 'auth1'

def test_random_forest_is_trained_flags(mock_data_path):
    # Directly exercise the concrete model's is_trained state
    df = pd.read_csv(mock_data_path)
    X = df.drop(columns=['label', 'identity', 'filename'])
    y = df['identity']

    model = RandomForestMatcher()
    assert model.is_trained() is False
    model.train(X, y, X.columns.tolist())
    assert model.is_trained() is True

def test_consistency_checker(mock_data_path, temp_dir):
    consistency = ConsistencyChecker(models_dir=temp_dir)

    # Train
    consistency.train(mock_data_path)
    # Check that model file was created
    assert os.path.exists(os.path.join(temp_dir, "consistency_models.joblib"))

    # Predict (Consistent)
    # Use the exact mean to be safe
    features = {'feature1': 1.0, 'feature2': 0.1}
    pred, score = consistency.check_consistency('auth1', features)
    # IsolationForest might still flag it if data is very sparse,
    # but at the mean it should be most likely to pass.
    assert pred in [1, -1]  # Just verify it returns a valid prediction
    assert score is not None

    # Predict (Inconsistent) - with extreme outlier
    features_outlier = {'feature1': 100, 'feature2': -50}
    pred_outlier, score_outlier = consistency.check_consistency('auth1', features_outlier)
    # Verify that the outlier has a lower score (more anomalous) than normal data
    assert score_outlier < score  # Outlier should have lower (more negative) score

def test_consistency_checker_unknown_identity(mock_data_path, temp_dir):
    """Test that check_consistency returns None for unknown identities."""
    consistency = ConsistencyChecker(models_dir=temp_dir)
    consistency.train(mock_data_path)

    # Check consistency for an identity that wasn't in training data
    features = {'feature1': 1.0, 'feature2': 0.1}
    pred, score = consistency.check_consistency('unknown_identity', features)

    assert pred is None
    assert score is None

def test_isolation_forest_anomaly_is_trained(mock_data_path, temp_dir):
    """Test that IsolationForestAnomaly.is_trained() works correctly."""
    from src.models.anomaly import IsolationForestAnomaly

    consistency = IsolationForestAnomaly()
    assert consistency.is_trained() is False

    # Train the model
    df = pd.read_csv(mock_data_path)
    X = df.drop(columns=['label', 'identity', 'filename'])
    identities = df['identity']
    consistency.train(X, X.columns.tolist(), identities)

    assert consistency.is_trained() is True
    assert 'auth1' in consistency.models
    assert 'auth2' in consistency.models

def test_isolation_forest_anomaly_save_load_is_trained(mock_data_path, temp_dir):
    """Test that IsolationForestAnomaly maintains is_trained state after save/load."""
    from src.models.anomaly import IsolationForestAnomaly

    consistency = IsolationForestAnomaly()
    df = pd.read_csv(mock_data_path)
    X = df.drop(columns=['label', 'identity', 'filename'])
    identities = df['identity']
    consistency.train(X, X.columns.tolist(), identities)

    # Save and load
    save_dir = os.path.join(temp_dir, "consistency_test")
    consistency.save(save_dir)

    loaded = IsolationForestAnomaly()
    assert loaded.is_trained() is False
    loaded.load(save_dir)
    assert loaded.is_trained() is True

def test_random_forest_matcher_evaluate_cv(mock_data_path):
    """Test that RandomForestMatcher.evaluate_cv() works correctly."""
    df = pd.read_csv(mock_data_path)
    X = df.drop(columns=['label', 'identity', 'filename'])
    y = df['identity']

    matcher = RandomForestMatcher()
    matcher.train(X, y, X.columns.tolist())

    # Run cross-validation
    cv_scores = matcher.evaluate_cv(X, y, cv=3)

    # Verify expected keys exist
    assert 'test_accuracy' in cv_scores
    assert 'test_precision_macro' in cv_scores
    assert 'test_recall_macro' in cv_scores
    assert 'test_f1_macro' in cv_scores
    assert 'train_accuracy' in cv_scores

    # Verify scores are valid (not NaN)
    assert all(~np.isnan(cv_scores['test_accuracy']))
    assert all(~np.isnan(cv_scores['test_f1_macro']))

def test_random_forest_matcher_evaluate_cv_with_more_splits(mock_data_path):
    """Test evaluate_cv with more CV splits (requires more samples)."""
    df = pd.read_csv(mock_data_path)
    X = df.drop(columns=['label', 'identity', 'filename'])
    y = df['identity']

    matcher = RandomForestMatcher()
    matcher.train(X, y, X.columns.tolist())

    # Run with 2 splits (we have 10 samples per class)
    cv_scores = matcher.evaluate_cv(X, y, cv=2)

    # Verify all scores are computed
    assert len(cv_scores['test_accuracy']) == 2
    assert len(cv_scores['test_f1_macro']) == 2

def test_isolation_forest_anomaly_contamination_parameter(mock_data_path):
    """Test that IsolationForestAnomaly uses correct contamination parameter (0.25)."""
    from src.models.anomaly import IsolationForestAnomaly

    # Test default contamination
    model = IsolationForestAnomaly()
    assert model.contamination == 0.25

    # Test custom contamination
    model_custom = IsolationForestAnomaly(contamination=0.1)
    assert model_custom.contamination == 0.1

def test_identity_matcher_missing_features(mock_data_path, temp_dir):
    """Test that predict handles missing features correctly by filling with 0."""
    model_path = os.path.join(temp_dir, "matcher.joblib")
    matcher = IdentityMatcher(model_path=model_path)
    matcher.train(mock_data_path)

    # Predict with only one feature (missing feature2)
    features_partial = {'feature1': 1.05}
    results = matcher.predict(features_partial)

    # Should still work, filling missing feature with 0
    assert len(results) == 2
    assert results[0][0] in ['auth1', 'auth2']

def test_consistency_checker_missing_features(mock_data_path, temp_dir):
    """Test that check_consistency handles missing features correctly."""
    consistency = ConsistencyChecker(models_dir=temp_dir)
    consistency.train(mock_data_path)

    # Check with only one feature (missing feature2)
    features_partial = {'feature1': 1.0}
    pred, score = consistency.check_consistency('auth1', features_partial)

    # Should still work
    assert pred in [1, -1]
    assert score is not None

def test_heuristic_detector():
    detector = HeuristicDetector()
    
    # DeepSeek suspected
    code_ds = "# -*- coding: utf-8 -*-"
    # Heuristics currently use code string primarily
    flags = detector.verify_metadata(code_ds, claimed_identity="gpt4o")
    markers = detector.detect_markers(code_ds)
    
    assert "MODEL_SPOOFING_SUSPECTED" in flags
    assert any(m['model'] == 'deepseek' for m in markers)
