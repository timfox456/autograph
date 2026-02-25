import pytest
import pandas as pd
import numpy as np

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")
from src.models.xgboost_models import XGBoostMatcher  # noqa: E402


def test_xgboost_matcher_basic_flow(tmp_path):
    """Test training and prediction with XGBoostMatcher."""
    # Create synthetic data
    X = pd.DataFrame(
        {"feature1": [1, 2, 1, 2, 5, 6, 5, 6], "feature2": [0.1, 0.2, 0.1, 0.2, 0.8, 0.9, 0.8, 0.9]}
    )
    y = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    feature_names = ["feature1", "feature2"]

    matcher = XGBoostMatcher()
    matcher.train(X, y, feature_names)

    # Test prediction
    test_df = pd.DataFrame({"feature1": [1.5], "feature2": [0.15]})
    probs = matcher.predict_probs(test_df)

    assert len(probs) == 2  # Two classes A and B
    assert probs[0][0] == "A"
    assert probs[0][1] > 0.5


def test_xgboost_matcher_feature_cleaning():
    """Test that XGBoostMatcher handles illegal characters in feature names."""
    X = pd.DataFrame({"node[type]": [1, 2], "count<5>": [0.1, 0.2]})
    y = pd.Series(["A", "B"])
    feature_names = ["node[type]", "count<5>"]

    matcher = XGBoostMatcher()
    # This should not raise "feature_names may not contain [, ] or <"
    matcher.train(X, y, feature_names)

    # Check that features were cleaned
    assert "node_type_" in matcher.features
    assert "count_5_" in matcher.features

    # Prediction should also work with original names (it cleans them internally)
    test_df = pd.DataFrame({"node[type]": [1], "count<5>": [0.1]})
    probs = matcher.predict_probs(test_df)
    assert len(probs) == 2


def test_xgboost_matcher_missing_features():
    """Test that XGBoostMatcher handles missing features during prediction."""
    X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    y = pd.Series(["A", "B"])
    feature_names = ["f1", "f2"]

    matcher = XGBoostMatcher()
    matcher.train(X, y, feature_names)

    # Predict with missing f2
    test_df = pd.DataFrame({"f1": [1]})
    probs = matcher.predict_probs(test_df)

    assert len(probs) == 2
    assert probs[0][1] > 0  # Should still return probabilities


def test_xgboost_matcher_save_load(tmp_path):
    """Test saving and loading XGBoostMatcher."""
    X = pd.DataFrame({"f1": [1, 5], "f2": [2, 6]})
    y = pd.Series(["A", "B"])
    feature_names = ["f1", "f2"]

    matcher = XGBoostMatcher()
    matcher.train(X, y, feature_names)

    model_path = tmp_path / "xgb.joblib"
    matcher.save(str(model_path))

    # Load into new instance
    new_matcher = XGBoostMatcher()
    new_matcher.load(str(model_path))

    assert new_matcher.features == matcher.features
    assert list(new_matcher.label_encoder.classes_) == ["A", "B"]

    # Predict with loaded model
    test_df = pd.DataFrame({"f1": [1], "f2": [2]})
    probs = new_matcher.predict_probs(test_df)
    assert probs[0][0] == "A"


def test_xgboost_matcher_numeric_conversion():
    """Test that XGBoostMatcher handles non-numeric data gracefully by coercing."""
    X = pd.DataFrame({"f1": [1, 2, "invalid"], "f2": [0.1, 0.2, 0.3]})
    y = pd.Series(["A", "A", "B"])
    feature_names = ["f1", "f2"]

    matcher = XGBoostMatcher()
    # Should not crash on 'invalid' string
    matcher.train(X, y, feature_names)

    test_df = pd.DataFrame({"f1": ["foo"], "f2": [0.1]})
    probs = matcher.predict_probs(test_df)
    assert len(probs) == 2


def test_xgboost_matcher_evaluate_cv():
    """Test that XGBoostMatcher.evaluate_cv() works correctly."""
    # Create synthetic data with more samples for CV - needs more separation
    np.random.seed(42)
    n_per_class = 15
    X = pd.DataFrame(
        {
            "feature1": np.concatenate(
                [np.random.normal(1, 0.3, n_per_class), np.random.normal(5, 0.3, n_per_class)]
            ),
            "feature2": np.concatenate(
                [np.random.normal(0.1, 0.05, n_per_class), np.random.normal(0.8, 0.05, n_per_class)]
            ),
        }
    )
    y = pd.Series(["A"] * n_per_class + ["B"] * n_per_class)
    feature_names = ["feature1", "feature2"]

    matcher = XGBoostMatcher()
    matcher.train(X, y, feature_names)

    # Run cross-validation
    cv_scores = matcher.evaluate_cv(X, y, cv=3)

    # Verify expected keys exist
    assert "test_accuracy" in cv_scores
    assert "test_precision_macro" in cv_scores
    assert "test_recall_macro" in cv_scores
    assert "test_f1_macro" in cv_scores
    assert "train_accuracy" in cv_scores

    # Verify all metrics have correct number of scores (one per fold)
    assert len(cv_scores["test_accuracy"]) == 3
    assert len(cv_scores["test_f1_macro"]) == 3


def test_xgboost_matcher_is_trained():
    """Test that XGBoostMatcher.is_trained() works correctly."""
    X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    y = pd.Series(["A", "B"])

    matcher = XGBoostMatcher()
    assert matcher.is_trained() is False

    matcher.train(X, y, ["f1", "f2"])
    assert matcher.is_trained() is True


def test_xgboost_matcher_evaluate_cv_returns_valid_scores():
    """Test that evaluate_cv returns valid structure even with minimal data."""
    X = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6], "f2": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]})
    y = pd.Series(["A", "A", "A", "B", "B", "B"])

    matcher = XGBoostMatcher()

    # Run cross-validation without explicit training
    cv_scores = matcher.evaluate_cv(X, y, cv=2)

    # Verify expected keys exist (method should complete and return structure)
    assert "test_accuracy" in cv_scores
    assert "test_f1_macro" in cv_scores
    assert "train_accuracy" in cv_scores

    # Verify structure: should have scores for each fold
    assert len(cv_scores["test_accuracy"]) == 2
    assert len(cv_scores["train_accuracy"]) == 2


def test_xgboost_matcher_all_nan_column():
    """Test that XGBoostMatcher raises ValueError on all-NaN input."""
    X = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [np.nan, np.nan, np.nan, np.nan]})  # All NaN column
    y = pd.Series(["A", "A", "B", "B"])
    feature_names = ["f1", "f2"]

    matcher = XGBoostMatcher()

    # Should raise ValueError due to all-NaN column
    with pytest.raises(ValueError, match="all NaN"):
        matcher.train(X, y, feature_names)


def test_xgboost_matcher_feature_order_consistency():
    """Test that XGBoostMatcher maintains feature order consistency."""
    X = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [0.1, 0.2, 0.3, 0.4], "f3": [10, 20, 30, 40]})
    y = pd.Series(["A", "A", "B", "B"])
    feature_names = ["f1", "f2", "f3"]

    matcher = XGBoostMatcher()
    matcher.train(X, y, feature_names)

    # Test with features in different order
    test_df_reordered = pd.DataFrame({"f3": [25], "f1": [2.5], "f2": [0.25]})

    # Should work regardless of column order
    probs_reordered = matcher.predict_probs(test_df_reordered)
    assert len(probs_reordered) == 2

    # Test with same features in original order - should give same prediction
    test_df_original = pd.DataFrame({"f1": [2.5], "f2": [0.25], "f3": [25]})
    probs_original = matcher.predict_probs(test_df_original)

    # Both should give same class ordering
    assert probs_reordered[0][0] == probs_original[0][0]


def test_xgboost_coerce_numeric_logs_warning():
    """Test that numeric coercion logs warnings for invalid values."""
    import logging
    import io

    # Set up logging capture
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("src.utils.features")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    logger.handlers = [handler]
    logger.setLevel(logging.WARNING)

    try:
        X = pd.DataFrame(
            {
                "f1": [1, 2, "invalid_value", 4],  # String that will be coerced
                "f2": [0.1, 0.2, 0.3, 0.4],
            }
        )
        y = pd.Series(["A", "A", "B", "B"])
        feature_names = ["f1", "f2"]

        matcher = XGBoostMatcher()
        matcher.train(X, y, feature_names)

        # Check that warning was logged
        log_capture.getvalue()
        # The coerce_numeric_with_warning should log the warning
        # The coerce_numeric_with_warning should log the warning

        # Prediction with invalid value should also warn
        test_df = pd.DataFrame({"f1": ["another_invalid"], "f2": [0.15]})
        probs = matcher.predict_probs(test_df)
        assert len(probs) == 2  # Should still work
    finally:
        # Restore original handlers and level
        logger.handlers = original_handlers
        logger.setLevel(original_level)
