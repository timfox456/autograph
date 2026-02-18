import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models.supervised import IdentityMatcher
from src.models.anomaly import ConsistencyChecker
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATASET_PATH = "autograph-ds/research/data/processed/dataset.csv"

def report():
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found. Please run process_dataset.py")
        return

    df = pd.read_csv(DATASET_PATH)
    
    # Filter out identities with only 1 sample as they can't be split for stratification
    counts = df['identity'].value_counts()
    valid_ids = counts[counts >= 2].index
    df = df[df['identity'].isin(valid_ids)]
    
    print(f"Dataset Size (Filtered for split): {len(df)}")
    print(f"Identities (Filtered for split): {df['identity'].nunique()}")
    print("-" * 30)

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['identity'])
    
    # Save temp train
    train_df.to_csv("train_temp.csv", index=False)
    
    # Identity Matcher
    matcher = IdentityMatcher()
    matcher.train("train_temp.csv")
    
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        features = row.drop(['label', 'identity', 'filename']).to_dict()
        preds = matcher.predict(features)
        y_true.append(row['identity'])
        y_pred.append(preds[0][0])
    
    print("\nIDENTITY MATCHER REPORT")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Consistency Checker
    consistency = ConsistencyChecker()
    consistency.train("train_temp.csv")
    
    positives = [] # Should be consistent
    negatives = [] # Should be anomalies
    
    # Positive samples (same identity)
    for _, row in test_df.iterrows():
        pred, _ = consistency.check_consistency(row['identity'], row.drop(['label', 'identity', 'filename']).to_dict())
        if pred is not None:
            positives.append(pred == 1)
            
    # Negative samples (wrong identity)
    identities = df['identity'].unique()
    for _, row in test_df.iterrows():
        other_ids = [i for i in identities if i != row['identity']]
        if other_ids:
            wrong_id = np.random.choice(other_ids)
            pred, _ = consistency.check_consistency(wrong_id, row.drop(['label', 'identity', 'filename']).to_dict())
            if pred is not None:
                negatives.append(pred == -1)
                
    print("\nCONSISTENCY CHECKER REPORT")
    print(f"Consistency Hit Rate (Correct Identity): {np.mean(positives):.4f}")
    print(f"Anomaly Detection Rate (Wrong Identity): {np.mean(negatives):.4f}")

    os.remove("train_temp.csv")

if __name__ == "__main__":
    report()
