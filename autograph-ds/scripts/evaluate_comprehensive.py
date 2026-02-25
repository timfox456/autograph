#!/usr/bin/env python3
"""Comprehensive model evaluation with per-class metrics."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from src.models.supervised import RandomForestMatcher
from src.models.xgboost_models import XGBoostMatcher
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, model_name, feature_names):
    """
    Comprehensive evaluation of a model.
    
    Returns dict with all metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    model.train(X_train, y_train, feature_names)
    
    probs_batch_for_pred = model.predict_probs_batch(X_test)
    y_pred_list = [probs[0][0] for probs in probs_batch_for_pred]
    y_pred = np.array(y_pred_list)
    
    probs_batch = model.predict_probs_batch(X_test)
    classes = np.unique(y_test)
    
    y_pred_proba = np.zeros((len(X_test), len(classes)))
    # Efficient dict-lookup per row instead of triple-nested loop
    for i, probs in enumerate(probs_batch):
        # Convert list of tuples to dict for O(1) lookup
        prob_dict = dict(probs)
        for j, cls in enumerate(classes):
            y_pred_proba[i, j] = prob_dict.get(cls, 0.0)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro): {recall_macro:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    
    per_class = {}
    
    for cls in classes:
        y_true_binary = (y_test == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        per_class[str(cls)] = {
            'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            'f1': float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
            'support': int((y_test == cls).sum())
        }
    
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    y_test_binarized = label_binarize(y_test, classes=classes)
    if len(classes) == 2:
        roc_auc = roc_auc_score(y_test_binarized, y_pred_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr', average='macro')
    
    print(f"\nROC-AUC (macro): {roc_auc:.4f}")
    
    return {
        'model_name': model_name,
        'overall': {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'roc_auc': float(roc_auc)
        },
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'classes': classes.tolist()
    }

def main():
    # Load the pre-saved train/test splits produced by train_models.py.
    # These apply the same identity-count filter (>=2 samples) used during
    # actual model training, ensuring this evaluation uses the exact same
    # holdout set the models were trained against.
    data_dir = Path("research/data/processed")
    train_df = pd.read_csv(data_dir / "dataset_train.csv")
    test_df = pd.read_csv(data_dir / "dataset_test.csv")

    meta_cols = ['label', 'identity', 'filename']
    feature_names = [c for c in train_df.columns if c not in meta_cols]

    X_train = train_df[feature_names]
    y_train = train_df['identity']
    X_test = test_df[feature_names]
    y_test = test_df['identity']

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Classes: {y_train.nunique()}")
    
    rf_model = RandomForestMatcher()
    rf_results = evaluate_model_comprehensive(
        rf_model, X_train, X_test, y_train, y_test, "RandomForestMatcher", feature_names
    )
    
    xgb_model = XGBoostMatcher()
    xgb_results = evaluate_model_comprehensive(
        xgb_model, X_train, X_test, y_train, y_test, "XGBoostMatcher", feature_names
    )
    
    results = {
        'random_forest': rf_results,
        'xgboost': xgb_results,
        'train_file': str(data_dir / "dataset_train.csv"),
        'test_file': str(data_dir / "dataset_test.csv"),
    }
    
    output_path = '.sisyphus/comprehensive_evaluation_results.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")
    
    print("\nModel Comparison:")
    print(f"{'Metric':<20} {'RandomForest':<15} {'XGBoost':<15}")
    print("-" * 50)
    for metric in ['accuracy', 'f1_macro', 'f1_weighted', 'roc_auc']:
        rf_val = rf_results['overall'][metric]
        xgb_val = xgb_results['overall'][metric]
        print(f"{metric:<20} {rf_val:<15.4f} {xgb_val:<15.4f}")

if __name__ == "__main__":
    main()
