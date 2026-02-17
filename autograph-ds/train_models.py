from src.models.supervised import IdentityMatcher
from src.models.anomaly import ConsistencyChecker
import os

def main():
    dataset_path = "autograph-ds/research/data/processed/dataset.csv"
    models_dir = "autograph-ds/research/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Train Supervised Matcher
    print("Training Supervised Matcher...")
    matcher = IdentityMatcher(model_path=os.path.join(models_dir, "matcher.joblib"))
    matcher.train(dataset_path)
    
    # Train Consistency Checker
    print("\nTraining Consistency Checker...")
    consistency = ConsistencyChecker(models_dir=models_dir)
    consistency.train(dataset_path)
    
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
