"""Feature selection utilities using sklearn SelectFromModel."""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


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
    # Fit RF to get feature importances
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced'
    )
    rf.fit(X, y)

    # Select top n_features
    selector = SelectFromModel(
        rf,
        max_features=n_features,
        threshold=-np.inf  # Select purely by max_features
    )
    X_selected = selector.fit_transform(X, y)

    # Get selected feature names
    if isinstance(X, pd.DataFrame):
        selected_features = X.columns[selector.get_support()].tolist()
    else:
        selected_features = [f'feature_{i}' for i in range(X_selected.shape[1])]

    return X_selected, selected_features, selector


def get_feature_importance_ranking(X, y, random_state=42):
    """
    Get feature importance ranking from RandomForest.

    Returns DataFrame with feature names and importance scores.
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced'
    )
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': X.columns if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])],
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df['cumulative'] = importance_df['importance'].cumsum()
    importance_df['cumulative_pct'] = importance_df['cumulative'] / importance_df['importance'].sum()

    return importance_df
