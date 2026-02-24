#!/usr/bin/env python3
"""Analyze current anomaly detection performance (TNR calculation)."""

import pandas as pd
import numpy as np
from pathlib import Path
from src.models.anomaly import IsolationForestAnomaly

def calculate_tnr(anomaly_model, test_df, feature_names):
    """
    Calculate True Negative Rate (TNR) for anomaly detection.
    
    TNR = correctly rejected cross-identity samples / total cross-identity samples
    
    For each identity pair (A, B) where A != B:
    - Test identity A's samples against identity B's model
    - Count how many are correctly rejected (predicted as -1/anomaly)
    """
    identities = test_df['identity'].unique()
    
    total_tests = 0
    correct_rejections = 0
    
    print(f"\nTesting {len(identities)} identities...")
    
    for target_identity in identities:
        if target_identity not in anomaly_model.models:
            continue
            
        # Get all samples from OTHER identities (cross-identity)
        cross_identity_df = test_df[test_df['identity'] != target_identity]
        
        if len(cross_identity_df) == 0:
            continue
        
        # Test each cross-identity sample
        for idx, row in cross_identity_df.iterrows():
            features = row[feature_names].values.reshape(1, -1)
            X_test = pd.DataFrame(features, columns=feature_names)
            
            prediction, score = anomaly_model.score(target_identity, X_test)
            
            if prediction is not None:
                total_tests += 1
                # prediction == -1 means anomaly (correctly rejected)
                if prediction == -1:
                    correct_rejections += 1
    
    if total_tests == 0:
        return 0.0, 0, 0
    
    tnr = correct_rejections / total_tests
    return tnr, correct_rejections, total_tests


def main():
    base = Path(__file__).parent
    dataset_path = base / "research/data/processed/dataset_329_features.csv"
    models_dir = base / "research/models"
    
    print("="*60)
    print("ANOMALY DETECTION PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Identities: {df['identity'].nunique()}")
    print(f"Identity distribution:")
    print(df['identity'].value_counts())
    
    # Extract features
    feature_cols = [col for col in df.columns if col not in ['label', 'identity', 'filename']]
    X = df[feature_cols]
    
    print(f"\nFeatures: {len(feature_cols)}")
    
    # Test different contamination values
    contamination_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 'auto']
    
    results = []
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT CONTAMINATION VALUES")
    print("="*60)
    
    for contamination in contamination_values:
        print(f"\n{'='*60}")
        print(f"Testing contamination={contamination}")
        print(f"{'='*60}")
        
        # Train anomaly detection model
        anomaly_model = IsolationForestAnomaly(
            contamination=contamination if contamination != 'auto' else 'auto',
            random_state=42
        )
        
        print(f"Training IsolationForest models for each identity...")
        anomaly_model.train(X, feature_cols, df['identity'])
        
        # Calculate TNR
        print(f"Calculating TNR (True Negative Rate)...")
        tnr, correct, total = calculate_tnr(anomaly_model, df, feature_cols)
        
        print(f"\nResults for contamination={contamination}:")
        print(f"  Total cross-identity tests: {total}")
        print(f"  Correct rejections: {correct}")
        print(f"  TNR (True Negative Rate): {tnr:.4f} ({tnr*100:.2f}%)")
        
        results.append({
            'contamination': str(contamination),
            'tnr': float(tnr),
            'correct_rejections': correct,
            'total_tests': total
        })
        
        # Check if we hit the target
        if tnr >= 0.80:
            print(f"  ✓ TARGET ACHIEVED: TNR >= 80%")
        else:
            print(f"  ✗ Below target: TNR < 80%")
    
    # Find best contamination
    best_result = max(results, key=lambda x: x['tnr'])
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nContamination vs TNR:")
    for r in results:
        status = "✓" if r['tnr'] >= 0.80 else "✗"
        print(f"  {status} contamination={r['contamination']:>6}: TNR={r['tnr']:.4f} ({r['tnr']*100:>5.2f}%)")
    
    print(f"\nBest configuration:")
    print(f"  Contamination: {best_result['contamination']}")
    print(f"  TNR: {best_result['tnr']:.4f} ({best_result['tnr']*100:.2f}%)")
    
    if best_result['tnr'] >= 0.80:
        print(f"\n✓ TARGET ACHIEVED: Best TNR >= 80%")
    else:
        print(f"\n✗ TARGET NOT ACHIEVED: Best TNR < 80%")
        print(f"  May need to try alternative anomaly detectors")
    
    # Save results
    import json
    results_path = base / ".sisyphus/anomaly_tuning_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'best_contamination': best_result['contamination'],
            'best_tnr': best_result['tnr'],
            'target_achieved': best_result['tnr'] >= 0.80
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
