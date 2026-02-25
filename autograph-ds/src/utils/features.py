"""Feature sanitization and validation utilities."""

import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sanitize_feature_names(feature_names: List[str]) -> List[str]:
    """
    Sanitizes feature names for model compatibility by replacing characters
    that some libraries disallow (e.g., XGBoost) and ensuring uniqueness.

    Specifically replaces: [ ] < > with underscores

    Args:
        feature_names: List of original feature names

    Returns:
        List of sanitized feature names with uniqueness enforced
    """
    clean_features: List[str] = []
    seen: dict[str, int] = {}
    for f in feature_names:
        clean = f.replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        clean_features.append(clean)
    return clean_features


def coerce_numeric_with_warning(
    df: pd.DataFrame, fill_value: float = 0.0, context: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert DataFrame to numeric types, logging warnings when coercion occurs.

    Unlike pd.to_numeric with errors='coerce', this function tracks which values
    failed conversion and logs them at the WARNING level.

    Args:
        df: DataFrame with potentially non-numeric values
        fill_value: Value to use for failed conversions (default: 0.0)
        context: Optional context string for error messages (e.g., "training", "prediction")

    Returns:
        DataFrame with all values converted to numeric, NaN filled with fill_value

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 'invalid']})
        >>> coerce_numeric_with_warning(df, context="feature 'a' in training")
        # Logs: WARNING - 1 value(s) coerced to NaN in feature 'a' in training
    """
    result = df.copy()
    context_str = f" in {context}" if context else ""

    for col in result.columns:
        original = result[col].copy()
        converted = pd.to_numeric(original, errors="coerce")

        # Check for coercion
        coerced_mask = original.notna() & converted.isna()
        num_coerced = coerced_mask.sum()

        if num_coerced > 0:
            # Get sample of coerced values
            coerced_values = original[coerced_mask].unique()
            sample_values = coerced_values[:5]  # Show up to 5 examples
            sample_str = ", ".join(str(v) for v in sample_values)
            if len(coerced_values) > 5:
                sample_str += f" (and {len(coerced_values) - 5} more unique values)"

            msg = (
                f"{num_coerced} value(s) coerced to NaN{context_str}, "
                f"column '{col}': {sample_str}"
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

        result[col] = converted.fillna(fill_value)

    return result


def validate_input_features(
    X: pd.DataFrame,
    expected_features: List[str],
    allow_extra: bool = True,
    context: Optional[str] = None,
) -> None:
    """
    Validate input features match expected schema.

    Args:
        X: Input DataFrame to validate
        expected_features: List of expected feature names
        allow_extra: If False, raises error when extra features are present
        context: Optional context for error messages

    Raises:
        ValueError: If validation fails
    """
    context_str = f" ({context})" if context else ""

    # Check for missing features
    missing = [f for f in expected_features if f not in X.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required features{context_str}: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}"
        )

    # Check for unexpected features
    if not allow_extra:
        extra = [f for f in X.columns if f not in expected_features]
        if extra:
            raise ValueError(
                f"Unexpected {len(extra)} extra features{context_str}: {extra[:10]}"
                f"{'...' if len(extra) > 10 else ''}"
            )

    # Check for NaN/inf values
    for col in expected_features:
        if col in X.columns:
            series = X[col]
            nan_count = series.isna().sum()
            if nan_count > 0:
                logger.warning(f"{nan_count} NaN value(s) in column '{col}'{context_str}")

            # Check for infinite values in numeric columns
            try:
                numeric_series = pd.to_numeric(series, errors="coerce")
                inf_count = np.isinf(numeric_series).sum()
                if inf_count > 0:
                    logger.warning(f"{inf_count} infinite value(s) in column '{col}'{context_str}")
            except (TypeError, ValueError):
                pass  # Non-numeric column, skip inf check


def validate_no_data_leakage(
    train_df: pd.DataFrame, test_df: pd.DataFrame, id_columns: Optional[List[str]] = None
) -> None:
    """
    Validate that there's no data leakage between train and test sets.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        id_columns: Columns to use for identifying duplicate samples

    Raises:
        ValueError: If overlapping samples are detected
    """
    if id_columns is None:
        # Use all columns as identifier
        id_columns = list(train_df.columns)

    # Check for exact row matches
    train_ids = set(tuple(row) for row in train_df[id_columns].values)
    test_ids = set(tuple(row) for row in test_df[id_columns].values)

    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} sample(s) appear in both "
            f"training and test sets based on columns {id_columns}"
        )
