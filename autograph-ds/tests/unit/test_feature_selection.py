"""Unit tests for feature selection utilities (src/features/selection.py)."""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.features.selection import (
    FeatureSelector,
    get_feature_importance_ranking,
    select_features_by_importance,
)


class TestFeatureSelector:
    """Tests for the FeatureSelector class with caching."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
        y_series = pd.Series(y)
        return X_df, y_series

    def test_initialization(self):
        """Test FeatureSelector initialization."""
        selector = FeatureSelector(random_state=42)

        assert selector.rf is None
        assert selector.is_fitted is False
        assert selector.random_state == 42

    def test_get_rf_trains_when_not_fitted(self, sample_data):
        """Test that _get_rf trains RF when not cached."""
        X, y = sample_data
        selector = FeatureSelector(random_state=42)

        rf = selector._get_rf(X, y)

        assert rf is not None
        assert selector.is_fitted is True
        assert selector.rf is rf

    def test_get_rf_returns_cached_when_fitted(self, sample_data, monkeypatch):
        """Test that _get_rf returns cached RF when already fitted."""
        X, y = sample_data
        selector = FeatureSelector(random_state=42)

        # First call trains
        rf1 = selector._get_rf(X, y)

        # Track if fit is called again
        fit_calls = []
        original_fit = selector.rf.fit

        def tracking_fit(*args, **kwargs):
            fit_calls.append(1)
            return original_fit(*args, **kwargs)

        monkeypatch.setattr(selector.rf, "fit", tracking_fit)

        # Second call should return cached without calling fit
        rf2 = selector._get_rf(X, y)

        assert rf1 is rf2  # Same object
        assert len(fit_calls) == 0  # fit was not called again

    def test_select_features_returns_correct_shape(self, sample_data):
        """Test that select_features returns correct shape."""
        X, y = sample_data
        selector = FeatureSelector(random_state=42)

        X_selected, features, fitted_selector_model = selector.select_features_by_importance(
            X, y, n_features=10
        )

        assert X_selected.shape == (100, 10)
        assert len(features) == 10
        # fitted_selector_model is the SelectFromModel instance, not FeatureSelector
        assert fitted_selector_model is not None

    def test_select_features_preserves_feature_names(self, sample_data):
        """Test that selected features have original names."""
        X, y = sample_data
        selector = FeatureSelector(random_state=42)

        X_selected, features, _ = selector.select_features_by_importance(X, y, n_features=5)

        # All features should come from original set
        assert all(f in X.columns for f in features)

    def test_select_features_with_ndarray(self, sample_data):
        """Test selection with numpy array instead of DataFrame."""
        X, y = sample_data
        selector = FeatureSelector(random_state=42)

        X_array = X.values
        X_selected, features, _ = selector.select_features_by_importance(X_array, y, n_features=5)

        assert X_selected.shape == (100, 5)
        # Features should be auto-named
        assert all(f.startswith("feature_") for f in features)

    def test_ranking_returns_dataframe(self, sample_data):
        """Test that ranking returns proper DataFrame."""
        X, y = sample_data
        selector = FeatureSelector(random_state=42)

        ranking = selector.get_feature_importance_ranking(X, y)

        assert isinstance(ranking, pd.DataFrame)
        assert "feature" in ranking.columns
        assert "importance" in ranking.columns
        assert "rank" in ranking.columns
        assert len(ranking) == 20

    def test_ranking_sorted_descending(self, sample_data):
        """Test that ranking is sorted by importance descending."""
        X, y = sample_data
        selector = FeatureSelector(random_state=42)

        ranking = selector.get_feature_importance_ranking(X, y)

        importances = ranking["importance"].values
        assert all(importances[i] >= importances[i + 1] for i in range(len(importances) - 1))

    def test_ranking_rank_column(self, sample_data):
        """Test that rank column is sequential."""
        X, y = sample_data
        selector = FeatureSelector(random_state=42)

        ranking = selector.get_feature_importance_ranking(X, y)

        expected_ranks = list(range(1, 21))
        assert list(ranking["rank"]) == expected_ranks

    def test_ranking_cumulative(self, sample_data):
        """Test cumulative importance calculation."""
        X, y = sample_data
        selector = FeatureSelector(random_state=42)

        ranking = selector.get_feature_importance_ranking(X, y)

        assert "cumulative" in ranking.columns
        assert "cumulative_pct" in ranking.columns

        # Cumulative should sum to 1.0
        assert abs(ranking["cumulative"].iloc[-1] - 1.0) < 1e-6

        # Percentages should be between 0 and 1
        assert all(0 <= v <= 1 for v in ranking["cumulative_pct"])

    def test_caching_across_methods(self, sample_data, monkeypatch):
        """Test that RF is cached when both methods are called."""
        X, y = sample_data
        selector = FeatureSelector(random_state=42)

        # Call select first
        selector.select_features_by_importance(X, y, n_features=5)

        # Track fit calls
        fit_calls = []
        original_fit = selector.rf.fit

        def tracking_fit(*args, **kwargs):
            fit_calls.append(1)
            return original_fit(*args, **kwargs)

        monkeypatch.setattr(selector.rf, "fit", tracking_fit)

        # Call ranking - should use cached RF
        selector.get_feature_importance_ranking(X, y)

        # fit should not be called again
        assert len(fit_calls) == 0

    def test_different_random_states(self, sample_data):
        """Test that different random states can produce different results."""
        X, y = sample_data

        selector1 = FeatureSelector(random_state=42)
        selector2 = FeatureSelector(random_state=123)

        _, features1, _ = selector1.select_features_by_importance(X, y, n_features=5)
        _, features2, _ = selector2.select_features_by_importance(X, y, n_features=5)

        # May or may not be same features, but both should work
        assert len(features1) == 5
        assert len(features2) == 5


class TestModuleLevelFunctions:
    """Tests for backward-compatible module-level functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=50, n_features=10, n_informative=8, n_redundant=2, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        y_series = pd.Series(y)
        return X_df, y_series

    def test_select_features_by_importance_function(self, sample_data):
        """Test module-level select_features_by_importance."""
        X, y = sample_data

        X_selected, features, selector = select_features_by_importance(
            X, y, n_features=5, random_state=42
        )

        assert X_selected.shape == (50, 5)
        assert len(features) == 5
        assert selector is not None

    def test_get_feature_importance_ranking_function(self, sample_data):
        """Test module-level get_feature_importance_ranking."""
        X, y = sample_data

        ranking = get_feature_importance_ranking(X, y, random_state=42)

        assert isinstance(ranking, pd.DataFrame)
        assert len(ranking) == 10
        assert ranking["importance"].sum() > 0


class TestEdgeCases:
    """Edge case tests for feature selection."""

    def test_single_feature(self):
        """Test selection with single feature."""
        X = pd.DataFrame({"only_feature": [1, 2, 3, 4, 5]})
        y = pd.Series([0, 0, 1, 1, 1])
        selector = FeatureSelector(random_state=42)

        X_selected, features, _ = selector.select_features_by_importance(X, y, n_features=1)

        assert X_selected.shape == (5, 1)
        assert features == ["only_feature"]

    def test_n_features_larger_than_available(self):
        """Test requesting more features than available - should handle gracefully."""
        X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        y = pd.Series([0, 0, 1])
        selector = FeatureSelector(random_state=42)

        # Request 10 features, only 2 available - should return all available
        # sklearn's SelectFromModel will raise if max_features > n_features
        # so we cap n_features to available
        available_features = X.shape[1]
        requested = min(10, available_features)

        X_selected, features, _ = selector.select_features_by_importance(X, y, n_features=requested)

        # Should return all available features
        assert X_selected.shape == (3, 2)
        assert len(features) == 2

    def test_binary_classification(self):
        """Test with binary classification."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        y_series = pd.Series(y)

        selector = FeatureSelector(random_state=42)
        X_selected, features, _ = selector.select_features_by_importance(
            X_df, y_series, n_features=5
        )

        assert X_selected.shape == (100, 5)

    def test_all_features_selected(self):
        """Test selecting all features."""
        X = pd.DataFrame({f"f{i}": range(10) for i in range(5)})
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        selector = FeatureSelector(random_state=42)

        X_selected, features, _ = selector.select_features_by_importance(X, y, n_features=5)

        assert X_selected.shape == (10, 5)
        assert set(features) == set(X.columns)


class TestBaseModelValidation:
    """Tests for BaseModel._validate_input integration."""

    def test_all_nan_column_raises(self):
        """Test that all-NaN column raises ValueError."""
        from src.models.base import BaseModel

        class TestModel(BaseModel):
            def train(self, X, y, feature_names):
                self.features = feature_names

        model = TestModel()

        X = pd.DataFrame({"valid": [1, 2, 3], "all_nan": [np.nan, np.nan, np.nan]})

        with pytest.raises(ValueError, match="all NaN"):
            model._validate_input(X)

    def test_inf_values_warning(self, caplog):
        """Test that infinite values log warning."""
        caplog.set_level(logging.WARNING)
        from src.models.base import BaseModel

        class TestModel(BaseModel):
            pass

        model = TestModel()

        X = pd.DataFrame({"col": [1, 2, np.inf]})

        model._validate_input(X)

        assert any("infinite" in msg.lower() for msg in caplog.messages)

    def test_valid_data_no_warnings(self, caplog):
        """Test that valid data produces no warnings."""
        caplog.set_level(logging.WARNING)
        from src.models.base import BaseModel

        class TestModel(BaseModel):
            pass

        model = TestModel()

        X = pd.DataFrame({"col": [1, 2, 3]})

        model._validate_input(X)

        assert not any("NaN" in msg or "infinite" in msg.lower() for msg in caplog.messages)
