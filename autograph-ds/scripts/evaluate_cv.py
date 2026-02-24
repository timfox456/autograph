#!/usr/bin/env python3
"""Cross-validation evaluation script for all models."""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from src.models.supervised import RandomForestMatcher
from src.models.xgboost_models import XGBoostMatcher


def evaluate_model_cv(model, X, y, model_name):
    """Run cross-validation and print metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    scores = model.evaluate_cv(X, y, cv=5)
    
    for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
        train_mean = scores[f'train_{metric}'].mean()
        train_std = scores[f'train_{metric}'].std()
        test_mean = scores[f'test_{metric}'].mean()
        test_std = scores[f'test_{metric}'].std()
        
        print(f"{metric:20s} - Train: {train_mean:.4f} ± {train_std:.4f}  |  Val: {test_mean:.4f} ± {test_std:.4f}")
    
    return scores


def main():
    dataset_path = Path("research/data/processed/dataset.csv")
    df = pd.read_csv(dataset_path)
    
    X = df.drop(columns=['label', 'identity', 'filename'])
    y = df['identity']
    
    rf_model = RandomForestMatcher()
    rf_scores = evaluate_model_cv(rf_model, X, y, "RandomForestMatcher")
    
    xgb_model = XGBoostMatcher()
    xgb_scores = evaluate_model_cv(xgb_model, X, y, "XGBoostMatcher")
    
    results = {
        'random_forest': {k: v.tolist() for k, v in rf_scores.items()},
        'xgboost': {k: v.tolist() for k, v in xgb_scores.items()}
    }
    
    results_path = Path('.sisyphus/cv_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
