import pytest
import pandas as pd
import numpy as np
import os

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")
from src.models.ensemble import EnsembleMatcher  # noqa: E402


@pytest.fixture
def synthetic_data():
    """Create synthetic multi-class data with enough samples per class."""
    np.random.seed(42)
    n_per_class = 20
    X = pd.DataFrame({
        'feature1': np.concatenate([
            np.random.normal(1, 0.3, n_per_class),
            np.random.normal(5, 0.3, n_per_class),
            np.random.normal(9, 0.3, n_per_class),
        ]),
        'feature2': np.concatenate([
            np.random.normal(0.1, 0.05, n_per_class),
            np.random.normal(0.5, 0.05, n_per_class),
            np.random.normal(0.9, 0.05, n_per_class),
        ]),
    })
    y = pd.Series(['alice'] * n_per_class + ['bob'] * n_per_class + ['carol'] * n_per_class)
    feature_names = ['feature1', 'feature2']
    return X, y, feature_names


def test_ensemble_train_and_predict(synthetic_data):
    """Test basic train/predict flow."""
    X, y, feature_names = synthetic_data

    model = EnsembleMatcher()
    assert model.is_trained() is False

    model.train(X, y, feature_names)
    assert model.is_trained() is True

    # Predict a sample near class 'alice' centroid
    test_df = pd.DataFrame({'feature1': [1.0], 'feature2': [0.1]})
    probs = model.predict_probs(test_df)

    assert len(probs) == 3  # 3 classes
    assert probs[0][0] == 'alice'
    # Use a more stringent threshold than 0.5 for clear cases
    # Alice centroid is (1.0, 0.1) and test point is (1.0, 0.1)
    assert probs[0][1] > 0.85, f"Expected high confidence for 'alice', got {probs[0][1]:.3f}"
    # Verify sorted order and probability sum
    assert probs[0][1] > probs[1][1] > probs[2][1]
    assert abs(sum(p for _, p in probs) - 1.0) < 1e-6


def test_ensemble_predict_probs_batch(synthetic_data):
    """Test batch prediction returns correct shape."""
    X, y, feature_names = synthetic_data

    model = EnsembleMatcher()
    model.train(X, y, feature_names)

    test_df = pd.DataFrame({
        'feature1': [1.0, 5.0, 9.0],
        'feature2': [0.1, 0.5, 0.9],
    })
    batch_probs = model.predict_probs_batch(test_df)

    assert len(batch_probs) == 3  # 3 samples
    for sample_probs in batch_probs:
        assert len(sample_probs) == 3  # 3 classes each


def test_ensemble_predict_before_train():
    """Test that predicting before training raises ValueError."""
    model = EnsembleMatcher()
    test_df = pd.DataFrame({'feature1': [1.0], 'feature2': [0.1]})

    with pytest.raises(ValueError, match="not trained"):
        model.predict_probs(test_df)


def test_ensemble_save_load(synthetic_data, tmp_path):
    """Test save/load roundtrip preserves model behavior."""
    X, y, feature_names = synthetic_data

    model = EnsembleMatcher()
    model.train(X, y, feature_names)

    # Get predictions before save
    test_df = pd.DataFrame({'feature1': [1.0], 'feature2': [0.1]})
    probs_before = model.predict_probs(test_df)

    # Save and load
    model_path = str(tmp_path / "ensemble.joblib")
    model.save(model_path)
    assert os.path.exists(model_path)

    loaded_model = EnsembleMatcher()
    loaded_model.load(model_path)
    assert loaded_model.is_trained() is True
    assert loaded_model.features == feature_names

    # Predictions should match
    probs_after = loaded_model.predict_probs(test_df)
    assert probs_before[0][0] == probs_after[0][0]
    assert abs(probs_before[0][1] - probs_after[0][1]) < 1e-6


def test_ensemble_missing_features(synthetic_data):
    """Test that prediction handles missing features by filling with 0."""
    X, y, feature_names = synthetic_data

    model = EnsembleMatcher()
    model.train(X, y, feature_names)

    # Predict with only one feature (missing feature2)
    test_df = pd.DataFrame({'feature1': [1.0]})
    probs = model.predict_probs(test_df)

    # Should still return valid probabilities
    assert len(probs) == 3
    total_prob = sum(p for _, p in probs)
    assert abs(total_prob - 1.0) < 1e-6


def test_ensemble_without_param_files(synthetic_data):
    """Test that ensemble works without JSON parameter files (uses defaults)."""
    X, y, feature_names = synthetic_data

    # No param files — should use default hyperparameters
    model = EnsembleMatcher(rf_params_path=None, xgb_params_path=None)
    model.train(X, y, feature_names)

    test_df = pd.DataFrame({'feature1': [5.0], 'feature2': [0.5]})
    probs = model.predict_probs(test_df)

    assert len(probs) == 3
    assert probs[0][0] == 'bob'



def test_ensemble_voting_weights_sum_to_one(synthetic_data):
    """Test that ensemble predictions sum to 1 (soft voting property)."""
    X, y, feature_names = synthetic_data
    
    model = EnsembleMatcher()
    model.train(X, y, feature_names)
    
    # Test multiple samples
    test_df = pd.DataFrame({
        'feature1': [1.0, 5.0, 9.0],
        'feature2': [0.1, 0.5, 0.9]
    })
    batch_probs = model.predict_probs_batch(test_df)
    
    for sample_probs in batch_probs:
        # Probabilities must sum to 1
        total = sum(p for _, p in sample_probs)
        assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, not 1.0"
        
        # Must have exactly 3 classes
        assert len(sample_probs) == 3
        
        # Must be sorted descending
        for i in range(len(sample_probs) - 1):
            assert sample_probs[i][1] >= sample_probs[i+1][1]


def test_ensemble_extract_classifier_params_no_model_type():
    """Test that _extract_classifier_params works without model_type parameter."""
    model = EnsembleMatcher()
    
    # Test with classifier__ prefixed params
    params = {
        'classifier__n_estimators': 100,
        'classifier__max_depth': 10,
        'other_param': 'ignored'
    }
    
    extracted = model._extract_classifier_params(params)
    
    assert 'n_estimators' in extracted
    assert 'max_depth' in extracted
    assert 'other_param' not in extracted  # Should be filtered out
    assert extracted['n_estimators'] == 100
    assert extracted['max_depth'] == 10
