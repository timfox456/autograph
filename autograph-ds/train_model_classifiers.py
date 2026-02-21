#!/usr/bin/env python3
"""Train classifiers on model embeddings for authorship verification.

This script trains multiple classifiers (RandomForest, SVM, kNN) on the
embeddings extracted by extract_model_embeddings.py for each model.

Usage:
    python train_model_classifiers.py [--models MODEL1 MODEL2 ...]

Options:
    --models    Which models to train classifiers for (default: all available)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from src.model_analysis.cache_manager import EmbeddingCache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_CACHE_DIR = Path("research/model_analysis/data/embeddings")
MODELS_DIR = Path("research/model_analysis/models")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_embeddings_and_labels(model_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load cached embeddings and corresponding labels.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Tuple of (embeddings, encoded_labels, label_names)
    """
    cache = EmbeddingCache(EMBEDDING_CACHE_DIR)
    
    # Load embeddings
    embeddings = cache.load(model_name)
    if embeddings is None:
        logger.error(f"No cached embeddings found for {model_name}")
        return None, None, None
    
    # Load metadata
    metadata_path = EMBEDDING_CACHE_DIR / "samples_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    identities = metadata["identities"]
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(identities)
    label_names = label_encoder.classes_.tolist()
    
    logger.info(f"Loaded {model_name} embeddings: {embeddings.shape}")
    logger.info(f"Number of identities: {len(label_names)}")
    
    return embeddings, encoded_labels, label_names


def filter_by_sample_count(embeddings: np.ndarray, labels: np.ndarray, min_samples: int = 2) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Filter identities with at least min_samples.
    
    Args:
        embeddings: Embedding matrix
        labels: Encoded labels
        min_samples: Minimum samples required per identity
    
    Returns:
        Tuple of (filtered_embeddings, filtered_labels, kept_indices)
    """
    from collections import Counter
    
    label_counts = Counter(labels)
    
    # Find labels with enough samples
    valid_labels = [label for label, count in label_counts.items() if count >= min_samples]
    
    # Filter samples
    mask = np.isin(labels, valid_labels)
    filtered_embeddings = embeddings[mask]
    filtered_labels = labels[mask]
    kept_indices = np.where(mask)[0].tolist()
    
    logger.info(f"Filtered from {len(labels)} to {len(filtered_labels)} samples "
                f"({len(valid_labels)}/{len(label_counts)} identities retained)")
    
    return filtered_embeddings, filtered_labels, kept_indices


def train_classifiers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    model_name: str
) -> Dict:
    """Train classifiers on embeddings.
    
    Args:
        embeddings: Embedding matrix
        labels: Encoded labels
        label_names: List of label names
        model_name: Name of the embedding model
    
    Returns:
        Dictionary with results for each classifier
    """
    results = {}
    
    # Filter identities with at least 2 samples
    embeddings, labels, _ = filter_by_sample_count(embeddings, labels, min_samples=2)
    
    if len(np.unique(labels)) < 2:
        logger.warning(f"Not enough identities with >=2 samples for {model_name}")
        return results
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Define classifiers
    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "SVM": SVC(
            kernel='rbf',
            C=1.0,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        "kNN": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
    }
    
    # Train and evaluate each classifier
    for clf_name, clf in classifiers.items():
        logger.info(f"Training {clf_name} on {model_name} embeddings...")
        
        try:
            # Train
            clf.fit(X_train, y_train)
            
            # Predict
            y_pred = clf.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=0
            )
            
            results[clf_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "classifier": clf,
            }
            
            logger.info(f"  {clf_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train {clf_name}: {e}")
            continue
    
    return results


def save_model(results: Dict, model_name: str):
    """Save trained classifiers to disk.
    
    Args:
        results: Dictionary with classifier results
        model_name: Name of the embedding model
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for clf_name, result in results.items():
        if "classifier" in result:
            save_path = MODELS_DIR / f"{model_name}_{clf_name.lower()}.joblib"
            joblib.dump(result["classifier"], save_path)
            logger.info(f"Saved {model_name} + {clf_name} model to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train classifiers on model embeddings for authorship verification"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Which models to train classifiers for (default: all with cached embeddings)"
    )
    
    args = parser.parse_args()
    
    # Determine which models to process
    cache = EmbeddingCache(EMBEDDING_CACHE_DIR)
    available_models = cache.list_cached_models()
    
    if args.models:
        models = [m for m in args.models if m in available_models]
        if not models:
            logger.error(f"None of the specified models have cached embeddings")
            return 1
    else:
        models = available_models
    
    if not models:
        logger.error("No cached embeddings found. Run extract_model_embeddings.py first.")
        return 1
    
    logger.info(f"Training classifiers for: {', '.join(models)}")
    
    # Train classifiers for each model
    all_results = {}
    
    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {model_name}")
        logger.info(f"{'='*60}")
        
        # Load embeddings
        embeddings, labels, label_names = load_embeddings_and_labels(model_name)
        if embeddings is None:
            continue
        
        # Train classifiers
        results = train_classifiers(embeddings, labels, label_names, model_name)
        
        if results:
            # Save models
            save_model(results, model_name)
            all_results[model_name] = results
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    
    for model_name, results in all_results.items():
        logger.info(f"\n{model_name}:")
        for clf_name, metrics in results.items():
            logger.info(f"  {clf_name:15s} Acc: {metrics['accuracy']:.4f}  "
                       f"F1: {metrics['f1']:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
