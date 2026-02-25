"""Utility modules for model preprocessing and pipeline creation."""

from .pipeline import (
    create_classification_pipeline,
    create_anomaly_pipeline,
    get_scaler_statistics,
)
from .features import sanitize_feature_names, coerce_numeric_with_warning, validate_input_features, validate_no_data_leakage

__all__ = [
    'create_classification_pipeline',
    'create_anomaly_pipeline',
    'get_scaler_statistics',
    'flatten_dna',
    'sanitize_feature_names',
    'coerce_numeric_with_warning',
    'validate_input_features',
    'validate_no_data_leakage',
]
# Import flatten_dna from parent utils module
# This is done by importing from the parent package's utils.py
import importlib.util
import os

parent_utils_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils.py')
spec = importlib.util.spec_from_file_location("_parent_utils", parent_utils_path)
_parent_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_parent_utils)
flatten_dna = _parent_utils.flatten_dna

__all__ = [
    'create_classification_pipeline',
    'create_anomaly_pipeline',
    'get_scaler_statistics',
    'flatten_dna',
]
