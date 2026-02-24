#!/usr/bin/env python3
"""Quick anomaly detection analysis with sampling for speed."""

import pandas as pd
import numpy as np
from pathlib import Path
from src.models.anomaly import IsolationForestAnomaly
import json

def calculate_tnr_sampled(anomaly_model, test_df, feature_names, max_samples_per_pair=10):
    """Calculate TNR with sampling for speed."""
    identities = test_df['identity'].unique()
    
    total_tests = 0
    correct_rejections = 0
    
    for target_identity in identities:
        if target_identity not in anomaly_model.models:
            continue
            
        # Get samples from OTHER identities (cross-identity)
        cross_identity_df = test_df[test_df['identity'] != target_identity]
        
        if len(cross_identity_df) == 0:
            continue
        
        # Sample for speed
        if len(cross_identity_df) > max_samples_per_pair:
            cross_identity_df = cross_identity_df.sample(max_samples_per_pair, random_state=42)
        
        # Test samples
        for idx, row in cross_identity_df.iterrows():
            features = row[feature_names].values.reshape(1, -1)
            X_test = pd.DataFrame(features, columns=feature_names)
            
            prediction, score = anomaly_model.score(target_identity, X_test)
            
            if prediction is not None:
                total_tests += 1
                if prediction == -1:  # Correctly rejected
                    correct_rejections += 1
    
    if total_tests == 0:
        return 0.0, 0, 0
    
    tnr = correct_rejections / total_tests
    return tnr, correct_rejections, total_tests


def main():
    base = Path(__file__).parent
    dataset_path = base / "research/data/processed/dataset_329_features.csv"
    
    print("="*60)
    print("QUICK ANOMALY DETECTION ANALYSIS")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset: {df.shape[0]} samples, {df['identity'].nunique()} identities")
    
    # Extract features
    feature_cols = [col for col in df.columns if col not in ['label', 'identity', 'filename']]
    X = df[feature_cols]
    
    # Test key contamination values
    contamination_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    results = []
    
    print("\n" + "="*60)
    print("TESTING CONTAMINATION VALUES (sampled)")
    print("="*60)
    
    for contamination in contamination_values:
        print(f"\nTesting contamination={contamination}...", end=" ")
        
        # Train model
        anomaly_model = IsolationForestAnomaly(
            contamination=contamination,
            random_state=42
        )
        anomaly_model.train(X, feature_cols, df['identity'])
        
        # Calculate TNR (sampled)
        tnr, correct, total = calculate_tnr_sampled(anomaly_model, df, feature_cols)
        
        print(f"TNR={tnr:.2%} ({correct}/{total})")
        
        results.append({
            'contamination': contamination,
            'tnr': float(tnr),
            'correct': correct,
            'total': total
        })
    
    # Find best
    best = max(results, key=lambda x: x['tnr'])
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBest contamination: {best['contamination']}")
    print(f"Best TNR: {best['tnr']:.2%}")
    
    if best['tnr'] >= 0.80:
        print("✓ TARGET ACHIEVED: TNR >= 80%")
    else:
        print("✗ Below target. Current best:", f"{best['tnr']:.2%}")
        print("  Target: 80%")
        print("  Gap:", f"{0.80 - best['tnr']:.2%}")
    
    # Save results
    results_path = base / ".sisyphus/anomaly_analysis_quick.json"
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'best_contamination': best['contamination'],
            'best_tnr': best['tnr'],
            'target_achieved': best['tnr'] >= 0.80,
            'note': 'Sampled analysis for speed'
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
