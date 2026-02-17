import pytest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from src.models.supervised import IdentityMatcher
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
    assert pred == 1
    
    # Predict (Inconsistent)
    features_outlier = {'feature1': 100, 'feature2': -50}
    pred_outlier, score_outlier = consistency.check_consistency('auth1', features_outlier)
    assert pred_outlier == -1

def test_heuristic_detector():
    detector = HeuristicDetector()
    
    # DeepSeek suspected
    code_ds = "# -*- coding: utf-8 -*-"
    # Heuristics currently use code string primarily
    flags = detector.verify_metadata(code_ds, claimed_identity="gpt4o")
    markers = detector.detect_markers(code_ds)
    
    assert "MODEL_SPOOFING_SUSPECTED" in flags
    assert any(m['model'] == 'deepseek' for m in markers)
