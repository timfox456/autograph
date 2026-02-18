from src.models.supervised import IdentityMatcher
from src.models.anomaly import ConsistencyChecker
from pathlib import Path
import os

def main():
    base = Path(__file__).parent
    dataset_path = base / "research/data/processed/dataset.csv"
    models_dir = base / "research/models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train Supervised Matcher
    print("Training Supervised Matcher...")
    matcher = IdentityMatcher(model_path=str(models_dir / "matcher.joblib"))
    matcher.train(str(dataset_path))
    
    # Train Consistency Checker
    print("\nTraining Consistency Checker...")
    consistency = ConsistencyChecker(models_dir=str(models_dir))
    consistency.train(str(dataset_path))
    
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
