"""Pipeline utilities for model preprocessing."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_classification_pipeline(classifier, use_scaler=True):
    """
    Create a sklearn Pipeline with optional StandardScaler.
    
    Args:
        classifier: The classifier model (e.g., RandomForest, XGBoost)
        use_scaler: Whether to include StandardScaler (default: True)
    
    Returns:
        Pipeline: Configured sklearn Pipeline
    
    Example:
        rf = RandomForestClassifier(n_estimators=100)
        pipeline = create_classification_pipeline(rf, use_scaler=True)
    """
    if use_scaler:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
    else:
        return Pipeline([
            ('classifier', classifier)
        ])


def create_anomaly_pipeline(detector, use_scaler=True):
    """
    Create a sklearn Pipeline for anomaly detection.
    
    Args:
        detector: The anomaly detector (e.g., IsolationForest)
        use_scaler: Whether to include StandardScaler (default: True)
    
    Returns:
        Pipeline: Configured sklearn Pipeline
    """
    if use_scaler:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('detector', detector)
        ])
    else:
        return Pipeline([
            ('detector', detector)
        ])


def get_scaler_statistics(pipeline, feature_names=None):
    """
    Extract scaler statistics from a fitted pipeline.
    
    Args:
        pipeline: Fitted Pipeline with StandardScaler
        feature_names: Optional list of feature names
    
    Returns:
        dict: Scaler mean and scale for each feature
    """
    if 'scaler' not in pipeline.named_steps:
        return None
    
    scaler = pipeline.named_steps['scaler']
    
    if not hasattr(scaler, 'mean_'):
        return None  # Scaler not fitted
    
    stats = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist()
    }
    
    if feature_names:
        stats['feature_names'] = feature_names
    
    return stats
