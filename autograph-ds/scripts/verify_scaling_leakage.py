#!/usr/bin/env python3
"""Verify no data leakage in scaling during cross-validation."""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.models.supervised import RandomForestMatcher
from src.models.xgboost_models import XGBoostMatcher


def verify_no_leakage():
    """Verify scaler is fit on training fold only."""
    print("=" * 70)
    print("SCALING LEAKAGE VERIFICATION")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv('research/data/processed/dataset_329_features.csv')
    X = df.drop(columns=['label', 'identity', 'filename'])
    y = df['identity']
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of identities: {y.nunique()}")
    print(f"Identities: {sorted(y.unique())}")
    
    # Manual CV to inspect scaler statistics
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    scaler_stats = []
    
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION FOLD ANALYSIS")
    print("=" * 70)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create fresh model
        model = RandomForestMatcher()
        
        # Fit on training fold only
        model.train(X_train, y_train, X_train.columns.tolist())
        
        # Extract scaler statistics
        scaler = model.pipeline.named_steps['scaler']
        stats = {
            'fold': fold_idx,
            'mean': scaler.mean_.copy(),
            'scale': scaler.scale_.copy(),
            'var': scaler.var_.copy(),
            'train_samples': len(train_idx),
            'val_samples': len(val_idx)
        }
        scaler_stats.append(stats)
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train samples: {len(train_idx)}")
        print(f"  Validation samples: {len(val_idx)}")
        print(f"  Scaler mean (first 5 features): {scaler.mean_[:5]}")
        print(f"  Scaler scale (first 5 features): {scaler.scale_[:5]}")
        print(f"  Scaler var (first 5 features): {scaler.var_[:5]}")
    
    # Verify statistics differ across folds
    print("\n" + "=" * 70)
    print("LEAKAGE VERIFICATION RESULTS")
    print("=" * 70)
    
    # Compare fold 0 and fold 1 means
    mean_diff = np.abs(scaler_stats[0]['mean'] - scaler_stats[1]['mean'])
    max_mean_diff = np.max(mean_diff)
    mean_diff_avg = np.mean(mean_diff)
    
    print(f"\nMean Statistics Comparison (Fold 0 vs Fold 1):")
    print(f"  Max difference: {max_mean_diff:.6f}")
    print(f"  Average difference: {mean_diff_avg:.6f}")
    print(f"  Min difference: {np.min(mean_diff):.6f}")
    
    # Verify scale differences
    scale_diff = np.abs(scaler_stats[0]['scale'] - scaler_stats[1]['scale'])
    max_scale_diff = np.max(scale_diff)
    scale_diff_avg = np.mean(scale_diff)
    
    print(f"\nScale Statistics Comparison (Fold 0 vs Fold 1):")
    print(f"  Max difference: {max_scale_diff:.6f}")
    print(f"  Average difference: {scale_diff_avg:.6f}")
    print(f"  Min difference: {np.min(scale_diff):.6f}")
    
    # Verify variance differences
    var_diff = np.abs(scaler_stats[0]['var'] - scaler_stats[1]['var'])
    max_var_diff = np.max(var_diff)
    var_diff_avg = np.mean(var_diff)
    
    print(f"\nVariance Statistics Comparison (Fold 0 vs Fold 1):")
    print(f"  Max difference: {max_var_diff:.6f}")
    print(f"  Average difference: {var_diff_avg:.6f}")
    print(f"  Min difference: {np.min(var_diff):.6f}")
    
    # Determine pass/fail
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    threshold = 0.001
    mean_pass = max_mean_diff > threshold
    scale_pass = max_scale_diff > threshold
    var_pass = max_var_diff > threshold
    
    if mean_pass:
        print(f"✅ PASS: Mean statistics differ across folds (max diff: {max_mean_diff:.6f})")
        print("   → Scaler is refit on each training fold (no leakage)")
    else:
        print(f"❌ FAIL: Mean statistics too similar (max diff: {max_mean_diff:.6f})")
        print("   → Possible data leakage detected")
    
    if scale_pass:
        print(f"✅ PASS: Scale statistics differ across folds (max diff: {max_scale_diff:.6f})")
    else:
        print(f"❌ FAIL: Scale statistics too similar (max diff: {max_scale_diff:.6f})")
    
    if var_pass:
        print(f"✅ PASS: Variance statistics differ across folds (max diff: {max_var_diff:.6f})")
    else:
        print(f"❌ FAIL: Variance statistics too similar (max diff: {max_var_diff:.6f})")
    
    overall_pass = mean_pass and scale_pass and var_pass
    
    print("\n" + "=" * 70)
    print(f"OVERALL RESULT: {'✅ PASS - NO LEAKAGE DETECTED' if overall_pass else '❌ FAIL - LEAKAGE DETECTED'}")
    print("=" * 70)
    
    # Additional verification: Compare fold 0 and fold 2
    print("\n" + "=" * 70)
    print("ADDITIONAL VERIFICATION (Fold 0 vs Fold 2)")
    print("=" * 70)
    
    mean_diff_02 = np.abs(scaler_stats[0]['mean'] - scaler_stats[2]['mean'])
    max_mean_diff_02 = np.max(mean_diff_02)
    
    scale_diff_02 = np.abs(scaler_stats[0]['scale'] - scaler_stats[2]['scale'])
    max_scale_diff_02 = np.max(scale_diff_02)
    
    print(f"Max mean difference (Fold 0 vs Fold 2): {max_mean_diff_02:.6f}")
    print(f"Max scale difference (Fold 0 vs Fold 2): {max_scale_diff_02:.6f}")
    
    if max_mean_diff_02 > threshold and max_scale_diff_02 > threshold:
        print("✅ PASS: Statistics differ between different fold pairs")
    else:
        print("❌ FAIL: Statistics too similar between different fold pairs")
    
    return overall_pass


if __name__ == "__main__":
    result = verify_no_leakage()
    exit(0 if result else 1)
