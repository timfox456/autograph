#!/usr/bin/env python3
"""Generate confusion matrix visualizations."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_confusion_matrix(cm, classes, model_name, output_path):
    """
    Create and save confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix (2D array)
        classes: Class labels
        model_name: Name of the model
        output_path: Path to save PNG
    """
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.02, f'Accuracy: {accuracy:.4f}', 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

def main():
    # Load evaluation results
    with open('.sisyphus/comprehensive_evaluation_results.json') as f:
        results = json.load(f)
    
    # Create reports directory
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    print("Generating confusion matrix visualizations...")
    
    # RandomForest
    rf_cm = np.array(results['random_forest']['confusion_matrix'])
    rf_classes = results['random_forest']['classes']
    plot_confusion_matrix(
        rf_cm, rf_classes, 'RandomForestMatcher',
        reports_dir / 'confusion_matrix_rf.png'
    )
    
    # XGBoost
    xgb_cm = np.array(results['xgboost']['confusion_matrix'])
    xgb_classes = results['xgboost']['classes']
    plot_confusion_matrix(
        xgb_cm, xgb_classes, 'XGBoostMatcher',
        reports_dir / 'confusion_matrix_xgb.png'
    )
    
    print("\nConfusion matrices saved to reports/ directory")
    print("  - confusion_matrix_rf.png")
    print("  - confusion_matrix_xgb.png")

if __name__ == "__main__":
    main()
