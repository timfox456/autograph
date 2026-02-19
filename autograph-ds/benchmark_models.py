import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.models.supervised import RandomForestMatcher
from src.models.xgboost_models import XGBoostMatcher

def benchmark():
    dataset_path = Path("research/data/processed/dataset.csv")
    if not dataset_path.exists():
        print("Dataset not found. Please run process_dataset.py first.")
        return

    df = pd.read_csv(dataset_path)
    counts = df['identity'].value_counts()
    valid_ids = counts[counts >= 2].index
    df = df[df['identity'].isin(valid_ids)]

    X = df.drop(columns=['label', 'identity', 'filename'])
    y = df['identity']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "RandomForest": RandomForestMatcher(n_estimators=100),
        "XGBoost": XGBoostMatcher()
    }

    results = []
    for name, model in models.items():
        print(f"Benchmarking {name}...")
        try:
            start_time = time.time()
            model.train(X_train, y_train, feature_names)
            train_time = time.time() - start_time

            y_pred = []
            start_time = time.time()
            for i in range(len(X_test)):
                row_df = X_test.iloc[[i]]
                probs = model.predict_probs(row_df)
                y_pred.append(probs[0][0])
            predict_time = (time.time() - start_time) / len(X_test)

            acc = accuracy_score(y_test, y_pred)
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Train Time (s)": train_time,
                "Predict Time/Sample (s)": predict_time
            })
        except Exception as e:
            print(f"Failed to benchmark {name}: {e}")

    res_df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    benchmark()
