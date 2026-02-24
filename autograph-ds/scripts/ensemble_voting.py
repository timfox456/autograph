import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.models.ensemble import EnsembleMatcher
from src.models.supervised import RandomForestMatcher
from src.models.xgboost_models import XGBoostMatcher


def main():
    base = Path(__file__).parent
    dataset_path = base / "research/data/processed/dataset_329_features.csv"
    sisyphus_dir = base / ".sisyphus"
    sisyphus_dir.mkdir(parents=True, exist_ok=True)
    
    rf_params_path = sisyphus_dir / "best_params_rf.json"
    xgb_params_path = sisyphus_dir / "best_params_xgb.json"
    
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {df['identity'].value_counts().to_dict()}")
    
    X = df.drop(columns=['label', 'identity', 'filename'])
    y = df['identity']
    feature_names = X.columns.tolist()
    
    print(f"\nFeatures: {len(feature_names)}")
    print(f"Samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    print("\n" + "="*60)
    print("Training Individual Models")
    print("="*60)
    
    print("\nTraining RandomForestMatcher...")
    rf_matcher = RandomForestMatcher()
    rf_matcher.train(X_train, y_train, feature_names)
    rf_preds = rf_matcher.predict(X_test)
    rf_probs = rf_matcher.predict_probs_batch(X_test)
    
    rf_accuracy = accuracy_score(y_test, rf_preds)
    rf_f1_macro = f1_score(y_test, rf_preds, average='macro')
    rf_f1_weighted = f1_score(y_test, rf_preds, average='weighted')
    
    print(f"RandomForest - Accuracy: {rf_accuracy:.4f}, F1_macro: {rf_f1_macro:.4f}, F1_weighted: {rf_f1_weighted:.4f}")
    
    print("\nTraining XGBoostMatcher...")
    xgb_matcher = XGBoostMatcher()
    xgb_matcher.train(X_train, y_train, feature_names)
    xgb_preds = xgb_matcher.predict(X_test)
    xgb_probs = xgb_matcher.predict_probs_batch(X_test)
    
    xgb_accuracy = accuracy_score(y_test, xgb_preds)
    xgb_f1_macro = f1_score(y_test, xgb_preds, average='macro')
    xgb_f1_weighted = f1_score(y_test, xgb_preds, average='weighted')
    
    print(f"XGBoost - Accuracy: {xgb_accuracy:.4f}, F1_macro: {xgb_f1_macro:.4f}, F1_weighted: {xgb_f1_weighted:.4f}")
    
    print("\n" + "="*60)
    print("Training Ensemble (VotingClassifier with Soft Voting)")
    print("="*60)
    
    print("\nTraining EnsembleMatcher...")
    ensemble = EnsembleMatcher(
        rf_params_path=str(rf_params_path),
        xgb_params_path=str(xgb_params_path),
        random_state=42
    )
    ensemble.train(X_train, y_train, feature_names)
    ensemble_preds = ensemble.predict(X_test)
    ensemble_probs = ensemble.predict_probs_batch(X_test)
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
    ensemble_f1_macro = f1_score(y_test, ensemble_preds, average='macro')
    ensemble_f1_weighted = f1_score(y_test, ensemble_preds, average='weighted')
    
    print(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}, F1_macro: {ensemble_f1_macro:.4f}, F1_weighted: {ensemble_f1_weighted:.4f}")
    
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)
    
    results = {
        "dataset": str(dataset_path),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": len(feature_names),
        "models": {
            "RandomForest": {
                "accuracy": float(rf_accuracy),
                "f1_macro": float(rf_f1_macro),
                "f1_weighted": float(rf_f1_weighted),
                "precision_macro": float(precision_score(y_test, rf_preds, average='macro')),
                "recall_macro": float(recall_score(y_test, rf_preds, average='macro'))
            },
            "XGBoost": {
                "accuracy": float(xgb_accuracy),
                "f1_macro": float(xgb_f1_macro),
                "f1_weighted": float(xgb_f1_weighted),
                "precision_macro": float(precision_score(y_test, xgb_preds, average='macro')),
                "recall_macro": float(recall_score(y_test, xgb_preds, average='macro'))
            },
            "Ensemble": {
                "accuracy": float(ensemble_accuracy),
                "f1_macro": float(ensemble_f1_macro),
                "f1_weighted": float(ensemble_f1_weighted),
                "precision_macro": float(precision_score(y_test, ensemble_preds, average='macro')),
                "recall_macro": float(recall_score(y_test, ensemble_preds, average='macro'))
            }
        },
        "improvements": {
            "ensemble_vs_rf_accuracy": float(ensemble_accuracy - rf_accuracy),
            "ensemble_vs_xgb_accuracy": float(ensemble_accuracy - xgb_accuracy),
            "ensemble_vs_rf_f1_weighted": float(ensemble_f1_weighted - rf_f1_weighted),
            "ensemble_vs_xgb_f1_weighted": float(ensemble_f1_weighted - xgb_f1_weighted)
        }
    }
    
    print("\nRandomForest:")
    print(f"  Accuracy: {rf_accuracy:.4f}")
    print(f"  F1 (macro): {rf_f1_macro:.4f}")
    print(f"  F1 (weighted): {rf_f1_weighted:.4f}")
    
    print("\nXGBoost:")
    print(f"  Accuracy: {xgb_accuracy:.4f}")
    print(f"  F1 (macro): {xgb_f1_macro:.4f}")
    print(f"  F1 (weighted): {xgb_f1_weighted:.4f}")
    
    print("\nEnsemble (Soft Voting):")
    print(f"  Accuracy: {ensemble_accuracy:.4f}")
    print(f"  F1 (macro): {ensemble_f1_macro:.4f}")
    print(f"  F1 (weighted): {ensemble_f1_weighted:.4f}")
    
    print("\nImprovement over individual models:")
    print(f"  vs RF Accuracy: {results['improvements']['ensemble_vs_rf_accuracy']:+.4f}")
    print(f"  vs XGB Accuracy: {results['improvements']['ensemble_vs_xgb_accuracy']:+.4f}")
    print(f"  vs RF F1 (weighted): {results['improvements']['ensemble_vs_rf_f1_weighted']:+.4f}")
    print(f"  vs XGB F1 (weighted): {results['improvements']['ensemble_vs_xgb_f1_weighted']:+.4f}")
    
    results_path = sisyphus_dir / "ensemble_voting_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    ensemble_model_path = sisyphus_dir / "ensemble_voting_model.joblib"
    ensemble.save(str(ensemble_model_path))
    print(f"Ensemble model saved to {ensemble_model_path}")
    
    print("\n" + "="*60)
    print("Verification: Loading and Testing Saved Model")
    print("="*60)
    
    ensemble_loaded = EnsembleMatcher()
    ensemble_loaded.load(str(ensemble_model_path))
    
    ensemble_loaded_preds = ensemble_loaded.predict(X_test)
    ensemble_loaded_accuracy = accuracy_score(y_test, ensemble_loaded_preds)
    
    print(f"\nLoaded ensemble accuracy: {ensemble_loaded_accuracy:.4f}")
    print(f"Original ensemble accuracy: {ensemble_accuracy:.4f}")
    print(f"Match: {np.allclose(ensemble_loaded_accuracy, ensemble_accuracy)}")
    
    if np.allclose(ensemble_loaded_accuracy, ensemble_accuracy):
        print("\n✓ Ensemble model successfully saved and loaded!")
    else:
        print("\n✗ Warning: Loaded model accuracy differs from original!")


if __name__ == "__main__":
    main()
