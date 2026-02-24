#!/usr/bin/env python3
"""Hyperparameter tuning for RandomForestMatcher using RandomizedSearchCV."""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config.hyperparameter_grids import RF_PARAM_GRID
import time

def tune_random_forest(dataset_path='research/data/processed/dataset_329_features.csv', 
                       n_iter=60, cv=5):
    """
    Tune RandomForest hyperparameters using RandomizedSearchCV.
    
    Args:
        dataset_path: Path to training dataset
        n_iter: Number of parameter settings sampled (default 60)
        cv: Number of CV folds (default 5)
    
    Returns:
        dict: Best parameters and best score
    """
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=['label', 'identity', 'filename'])
    y = df['identity']
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.nunique()} classes")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Setup RandomizedSearchCV
    print(f"\nStarting RandomizedSearchCV with n_iter={n_iter}, cv={cv}...")
    print(f"Parameter space size: {len(RF_PARAM_GRID)} parameters")
    print(f"Total combinations possible: {np.prod([len(v) for v in RF_PARAM_GRID.values()])}")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=RF_PARAM_GRID,
        n_iter=n_iter,
        cv=skf,
        scoring='f1_weighted',
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
        verbose=2,
        return_train_score=True
    )
    
    # Run search
    start_time = time.time()
    search.fit(X, y)
    elapsed = time.time() - start_time
    
    print(f"\nSearch completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # Extract results
    best_params = search.best_params_
    best_score = search.best_score_
    
    print(f"\nBest CV Score (f1_weighted): {best_score:.4f}")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Get top 5 configurations
    results_df = pd.DataFrame(search.cv_results_)
    top5 = results_df.nlargest(5, 'mean_test_score')[
        ['param_classifier__n_estimators', 
         'param_classifier__max_depth',
         'param_classifier__min_samples_split',
         'param_classifier__min_samples_leaf',
         'param_classifier__max_features',
         'param_classifier__bootstrap',
         'mean_test_score', 
         'std_test_score']
    ]
    
    print("\nTop 5 Configurations:")
    print(top5.to_string(index=False))
    
    # Save results
    output = {
        'best_params': best_params,
        'best_score': float(best_score),
        'n_iter': n_iter,
        'cv': cv,
        'scoring': 'f1_weighted',
        'dataset': dataset_path,
        'elapsed_seconds': elapsed,
        'top5_configs': top5.to_dict('records')
    }
    
    output_path = '.sisyphus/best_params_rf.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return output

if __name__ == "__main__":
    import sys
    
    # Allow command-line override of n_iter
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    
    results = tune_random_forest(n_iter=n_iter)
    
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
    print(f"Best F1 Score: {results['best_score']:.4f}")
    print(f"Best n_estimators: {results['best_params']['classifier__n_estimators']}")
    print(f"Best max_depth: {results['best_params']['classifier__max_depth']}")
