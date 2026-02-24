from src.models.xgboost_models import XGBoostMatcher
from src.models.anomaly import ConsistencyChecker
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    base = Path(__file__).parent
    dataset_path = base / "research/data/processed/dataset_329_features.csv"
    models_dir = base / "research/models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Filter out identities with only 1 sample as they can't be split for stratification
    counts = df['identity'].value_counts()
    valid_ids = counts[counts >= 2].index
    df = df[df['identity'].isin(valid_ids)]
    
    # Stratified train/test split (80/20)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['identity']
    )
    
    # Save splits to separate CSV files
    train_path = base / "research/data/processed/dataset_train.csv"
    test_path = base / "research/data/processed/dataset_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Extract features (drop metadata columns)
    X_train = train_df.drop(columns=['label', 'identity', 'filename'])
    y_train = train_df['identity']
    X_test = test_df.drop(columns=['label', 'identity', 'filename'])
    y_test = test_df['identity']
    feature_names = X_train.columns.tolist()
    
    # Train XGBoost Matcher
    # Uses tuned hyperparameters baked into the constructor defaults
    # (from RandomizedSearchCV: n_iter=60, best F1=0.9884)
    print("\nTraining XGBoost Matcher...")
    matcher = XGBoostMatcher()
    matcher.train(X_train, y_train, feature_names)
    
    # Evaluate on test set
    test_probs = matcher.predict_probs_batch(X_test)
    test_predictions = [probs[0][0] for probs in test_probs]
    test_accuracy = sum(pred == actual for pred, actual in zip(test_predictions, y_test)) / len(y_test)
    
    print(f"XGBoost Matcher - Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    matcher_path = models_dir / "matcher_xgb.joblib"
    matcher.save(str(matcher_path))
    print(f"XGBoost Matcher saved to {matcher_path}")
    
    # Train Consistency Checker
    print("\nTraining Consistency Checker...")
    consistency = ConsistencyChecker(models_dir=str(models_dir))
    consistency.train(str(train_path))
    
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
