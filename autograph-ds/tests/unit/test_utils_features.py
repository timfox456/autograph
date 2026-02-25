"""Unit tests for feature utilities (src/utils/features.py)."""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from src.utils.features import (
    coerce_numeric_with_warning,
    sanitize_feature_names,
    validate_input_features,
    validate_no_data_leakage,
)


class TestSanitizeFeatureNames:
    """Tests for sanitize_feature_names function."""

    def test_basic_sanitization(self):
        """Test basic character replacement."""
        features = ["node[type]", "count<5>", "range[0:10]", "val>min"]
        result = sanitize_feature_names(features)

        assert "node_type_" in result
        assert "count_5_" in result
        assert "range_0:10_" in result
        assert "val_min" in result

    def test_no_special_characters(self):
        """Test features without special characters remain unchanged."""
        features = ["feature1", "feature_2", "camelCase", "snake_case"]
        result = sanitize_feature_names(features)

        assert result == features

    def test_duplicate_handling(self):
        """Test that duplicate names after sanitization are made unique."""
        features = ["a[b]", "a<b>", "a_b"]  # All become 'a_b' or similar
        result = sanitize_feature_names(features)

        # All should be unique
        assert len(result) == len(set(result))

    def test_empty_list(self):
        """Test empty list returns empty list."""
        assert sanitize_feature_names([]) == []

    def test_preserves_order(self):
        """Test that order is preserved."""
        features = ["z[1]", "a[2]", "m[3]"]
        result = sanitize_feature_names(features)

        assert len(result) == 3
        assert result[0].startswith("z")
        assert result[1].startswith("a")
        assert result[2].startswith("m")


class TestCoerceNumericWithWarning:
    """Tests for coerce_numeric_with_warning function."""

    def test_valid_numeric_no_warning(self, caplog):
        """Test that valid numeric data produces no warnings."""
        caplog.set_level(logging.WARNING)

        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})

        result = coerce_numeric_with_warning(df, fill_value=0)

        assert "a" in result.columns
        assert "b" in result.columns
        # No coercion warnings
        assert not any("coerced to NaN" in msg for msg in caplog.messages)

    def test_coercion_logs_warning(self, caplog):
        """Test that coercion logs a warning."""
        caplog.set_level(logging.WARNING)

        df = pd.DataFrame({"col": [1, 2, "invalid", 4]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = coerce_numeric_with_warning(df, fill_value=0, context="test")

            # Check warning was raised
            assert len(w) >= 1
            assert any("coerced to NaN" in str(warning.message) for warning in w)

        # Check log message
        assert any("coerced to NaN" in msg for msg in caplog.messages)
        # Verify context appears in message
        assert any("test" in msg for msg in caplog.messages)

        # Invalid value should be filled with 0
        assert result["col"].iloc[2] == 0

    def test_fill_value_parameter(self):
        """Test custom fill value."""
        df = pd.DataFrame({"col": ["invalid"]})

        result = coerce_numeric_with_warning(df, fill_value=-999)

        assert result["col"].iloc[0] == -999

    def test_coercion_shows_sample_values(self, caplog):
        """Test that warning shows sample of coerced values."""
        caplog.set_level(logging.WARNING)

        df = pd.DataFrame({"col": ["bad1", "bad2", "bad3", "bad4", "bad5", "bad6"]})

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            coerce_numeric_with_warning(df, fill_value=0)

        # Should show sample of values in log
        log_text = " ".join(caplog.messages)
        assert "bad1" in log_text or "bad2" in log_text

    def test_mixed_types(self):
        """Test DataFrame with mixed types."""
        df = pd.DataFrame({"numeric": [1, 2, 3], "with_invalid": [1, "bad", 3]})

        result = coerce_numeric_with_warning(df, fill_value=0)

        assert list(result["numeric"]) == [1, 2, 3]
        assert result["with_invalid"].iloc[1] == 0

    def test_existing_nan_preserved(self):
        """Test that existing NaN values are filled."""
        df = pd.DataFrame({"col": [1.0, np.nan, 3.0]})

        result = coerce_numeric_with_warning(df, fill_value=-1)

        assert result["col"].iloc[1] == -1
        assert not result["col"].isna().any()


class TestValidateInputFeatures:
    """Tests for validate_input_features function."""

    def test_valid_input_no_errors(self):
        """Test that valid input passes validation."""
        X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        expected = ["f1", "f2"]

        # Should not raise
        validate_input_features(X, expected)

    def test_missing_features_raises(self):
        """Test that missing features raise ValueError."""
        X = pd.DataFrame({"f1": [1, 2, 3]})
        expected = ["f1", "f2"]

        with pytest.raises(ValueError, match="Missing"):
            validate_input_features(X, expected)

    def test_missing_features_shows_count(self):
        """Test that error message shows missing count."""
        X = pd.DataFrame({"f1": [1]})
        expected = ["f1", "f2", "f3", "f4"]

        with pytest.raises(ValueError) as exc:
            validate_input_features(X, expected)

        assert "3" in str(exc.value) or "Missing" in str(exc.value)

    def test_unexpected_features_raises_when_disabled(self):
        """Test that extra features raise when allow_extra=False."""
        X = pd.DataFrame({"f1": [1], "f2": [2], "f3": [3]})
        expected = ["f1", "f2"]

        with pytest.raises(ValueError, match="Unexpected"):
            validate_input_features(X, expected, allow_extra=False)

    def test_unexpected_features_allowed_by_default(self):
        """Test that extra features are allowed by default."""
        X = pd.DataFrame({"f1": [1], "f2": [2], "extra": [3]})
        expected = ["f1", "f2"]

        # Should not raise
        validate_input_features(X, expected)

    def test_nan_values_logged(self, caplog):
        """Test that NaN values are logged."""
        caplog.set_level(logging.WARNING)

        X = pd.DataFrame({"f1": [1, np.nan, 3]})
        expected = ["f1"]

        validate_input_features(X, expected)

        assert any("NaN" in msg for msg in caplog.messages)

    def test_context_in_error_message(self):
        """Test that context appears in error messages."""
        X = pd.DataFrame({"f1": [1]})
        expected = ["f1", "f2"]

        with pytest.raises(ValueError) as exc:
            validate_input_features(X, expected, context="prediction phase")

        assert "prediction phase" in str(exc.value)


class TestValidateNoDataLeakage:
    """Tests for validate_no_data_leakage function."""

    def test_no_overlap_passes(self):
        """Test that non-overlapping datasets pass."""
        train = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        test = pd.DataFrame({"id": [4, 5, 6], "value": ["d", "e", "f"]})

        # Should not raise
        validate_no_data_leakage(train, test, id_columns=["id"])

    def test_overlap_raises(self):
        """Test that overlapping samples raise ValueError."""
        train = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        test = pd.DataFrame({"id": [3, 4, 5], "value": ["c", "d", "e"]})  # 3 is in both

        with pytest.raises(ValueError, match="Data leakage"):
            validate_no_data_leakage(train, test, id_columns=["id"])

    def test_overlap_shows_count(self):
        """Test that error shows count of overlapping samples."""
        train = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        test = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})  # All overlap

        with pytest.raises(ValueError) as exc:
            validate_no_data_leakage(train, test, id_columns=["id"])

        assert "3" in str(exc.value)

    def test_default_all_columns(self):
        """Test that all columns are used when id_columns not specified."""
        train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        test = pd.DataFrame({"a": [1, 5], "b": [3, 6]})  # First row matches train[0]

        with pytest.raises(ValueError, match="Data leakage"):
            validate_no_data_leakage(train, test)

    def test_no_overlap_all_columns(self):
        """Test all-column check with non-overlapping data."""
        train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        test = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        # Should not raise
        validate_no_data_leakage(train, test)


class TestFeatureValidationEdgeCases:
    """Edge case tests for validation functions."""

    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        X = pd.DataFrame()
        expected = []

        # Should pass with empty expected
        validate_input_features(X, expected)

    def test_single_row(self):
        """Test validation with single row."""
        X = pd.DataFrame({"f1": [1]})
        expected = ["f1"]

        validate_input_features(X, expected)

    def test_many_missing_features_truncation(self):
        """Test that many missing features are truncated in message."""
        X = pd.DataFrame({"f1": [1]})
        expected = [f"f{i}" for i in range(20)]

        with pytest.raises(ValueError) as exc:
            validate_input_features(X, expected)

        msg = str(exc.value)
        # Should indicate more features not shown
        assert "..." in msg or len(msg) < 500
