"""Feature selection utilities using sklearn SelectFromModel."""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


class FeatureSelector:
    """
    Cached feature selector to avoid retraining RandomForest when both
    select_features_by_importance() and get_feature_importance_ranking() are called.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rf = None
        self.is_fitted = False

    def _get_rf(self, X, y):
        """Get or create fitted RandomForest, caching if already fitted."""
        if self.is_fitted and self.rf is not None:
            return self.rf

        self.rf = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, class_weight="balanced"
        )
        self.rf.fit(X, y)
        self.is_fitted = True
        return self.rf

    def select_features_by_importance(self, X, y, n_features=329):
        """
        Select top n_features based on RandomForest feature importance.

        Args:
            X: Feature matrix (DataFrame or ndarray)
            y: Target labels
            n_features: Number of features to select (default 329)

        Returns:
            X_selected: Selected feature matrix
            selected_features: List of selected feature names
            selector: Fitted SelectFromModel instance
        """
        # Get or train RF (cached)
        rf = self._get_rf(X, y)

        # Select top n_features
        selector = SelectFromModel(
            rf, max_features=n_features, threshold=-np.inf  # Select purely by max_features
        )
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        if isinstance(X, pd.DataFrame):
            selected_features = X.columns[selector.get_support()].tolist()
        else:
            selected_features = [f"feature_{i}" for i in range(X_selected.shape[1])]

        return X_selected, selected_features, selector

    def get_feature_importance_ranking(self, X, y):
        """
        Get feature importance ranking from RandomForest.

        Returns DataFrame with feature names and importance scores.

        NOTE: Uses cached RF if select_features_by_importance() was called first.
        """
        # Get or train RF (cached)
        rf = self._get_rf(X, y)

        importance_df = pd.DataFrame(
            {
                "feature": (
                    X.columns
                    if isinstance(X, pd.DataFrame)
                    else [f"feature_{i}" for i in range(X.shape[1])]
                ),
                "importance": rf.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        importance_df["rank"] = range(1, len(importance_df) + 1)
        importance_df["cumulative"] = importance_df["importance"].cumsum()
        importance_df["cumulative_pct"] = (
            importance_df["cumulative"] / importance_df["importance"].sum()
        )

        return importance_df


# Backward-compatible module-level functions that use the class internally
def select_features_by_importance(X, y, n_features=329, random_state=42):
    """
    Select top n_features based on RandomForest feature importance.

    Args:
        X: Feature matrix (DataFrame or ndarray)
        y: Target labels
        n_features: Number of features to select (default 329)
        random_state: Random seed for reproducibility

    Returns:
        X_selected: Selected feature matrix
        selected_features: List of selected feature names
        selector: Fitted SelectFromModel instance
    """
    selector = FeatureSelector(random_state=random_state)
    return selector.select_features_by_importance(X, y, n_features)


def get_feature_importance_ranking(X, y, random_state=42):
    """
    Get feature importance ranking from RandomForest.

    Returns DataFrame with feature names and importance scores.

    NOTE: This function trains a separate RandomForest. If you need to call
    both select_features_by_importance() and this function, use the
    FeatureSelector class directly to cache the fitted model.
    """
    selector = FeatureSelector(random_state=random_state)
    return selector.get_feature_importance_ranking(X, y)
