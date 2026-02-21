#!/usr/bin/env python3
"""Generate comparison report for model-based vs. Logical DNA authorship verification.

This script creates a comprehensive report comparing the performance of
model-based embeddings (CodeBERT, GraphCodeBERT, UniXcoder, CLAVE) against
the existing Logical DNA feature engineering approach.

Usage:
    python report_model_comparison.py [--output OUTPUT_PATH]

Options:
    --output    Path to save the report (default: research/model_analysis/reports/comparison_YYYYMMDD.md)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODELS_DIR = Path("research/model_analysis/models")
REPORTS_DIR = Path("research/model_analysis/reports")


def get_logical_dna_baseline() -> Dict:
    """Get baseline metrics from Logical DNA approach.
    
    Returns:
        Dictionary with baseline metrics
    """
    try:
        # Try to run report_metrics.py and capture output
        result = subprocess.run(
            ["python", "report_metrics.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse output for accuracy
        output = result.stdout
        baseline = {"accuracy": 0.0, "f1": 0.0, "note": "Baseline from Logical DNA"}
        
        # Try to extract accuracy from output
        for line in output.split('\n'):
            if 'accuracy' in line.lower() and '%' in line:
                try:
                    # Extract number before %
                    import re
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        baseline["accuracy"] = float(match.group(1)) / 100.0
                except:
                    pass
        
        return baseline
    
    except Exception as e:
        logger.warning(f"Could not get Logical DNA baseline: {e}")
        return {"accuracy": 0.0, "f1": 0.0, "note": "Baseline unavailable"}


def load_model_results() -> Dict:
    """Load results from trained models.
    
    Returns:
        Dictionary mapping model_name -> classifier_name -> metrics
    """
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from src.model_analysis.cache_manager import EmbeddingCache
    
    results = {}
    
    cache = EmbeddingCache("research/model_analysis/data/embeddings")
    available_models = cache.list_cached_models()
    
    for model_name in available_models:
        model_results = {}
        
        # Load embeddings
        embeddings = cache.load(model_name)
        if embeddings is None:
            continue
        
        # Load labels
        metadata_path = Path("research/model_analysis/data/embeddings") / "samples_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(metadata["identities"])
        
        # Filter identities with >=2 samples
        from collections import Counter
        label_counts = Counter(labels)
        valid_labels = [l for l, c in label_counts.items() if c >= 2]
        mask = np.isin(labels, valid_labels)
        embeddings = embeddings[mask]
        labels = labels[mask]
        
        if len(np.unique(labels)) < 2:
            continue
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Evaluate each classifier
        for clf_file in MODELS_DIR.glob(f"{model_name}_*.joblib"):
            clf_name = clf_file.stem.replace(f"{model_name}_", "").upper()
            
            try:
                clf = joblib.load(clf_file)
                y_pred = clf.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='macro', zero_division=0
                )
                
                model_results[clf_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            except Exception as e:
                logger.warning(f"Failed to evaluate {clf_file}: {e}")
        
        if model_results:
            results[model_name] = model_results
    
    return results


def generate_report(output_path: Optional[Path] = None) -> str:
    """Generate comparison report.
    
    Args:
        output_path: Path to save report (default: auto-generated)
    
    Returns:
        Report content as string
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = REPORTS_DIR / f"comparison_{timestamp}.md"
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get baseline
    baseline = get_logical_dna_baseline()
    
    # Get model results
    model_results = load_model_results()
    
    # Build report
    lines = []
    lines.append("# Model-Based Authorship Verification Comparison Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report compares the performance of pre-trained code models")
    lines.append("against the existing Logical DNA feature engineering approach for")
    lines.append("authorship verification on Python code.")
    lines.append("")
    
    # Find best performing model
    best_model = None
    best_accuracy = 0
    
    for model_name, classifiers in model_results.items():
        for clf_name, metrics in classifiers.items():
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_model = f"{model_name} + {clf_name}"
    
    if best_model:
        lines.append(f"**Best Model**: {best_model} (Accuracy: {best_accuracy:.2%})")
        lines.append("")
    
    # Comparison table
    lines.append("## Performance Comparison")
    lines.append("")
    lines.append("| Approach | Accuracy | Precision | Recall | F1 Score |")
    lines.append("|----------|----------|-----------|--------|----------|")
    
    # Baseline
    lines.append(f"| **Logical DNA (380 features)** | {baseline['accuracy']:.2%} | - | - | {baseline.get('f1', 0):.2%} |")
    lines.append("")
    
    # Model results
    for model_name, classifiers in sorted(model_results.items()):
        lines.append(f"**{model_name.upper()}**:")
        lines.append("")
        for clf_name, metrics in sorted(classifiers.items()):
            lines.append(f"| {model_name} + {clf_name} | "
                        f"{metrics['accuracy']:.2%} | "
                        f"{metrics['precision']:.2%} | "
                        f"{metrics['recall']:.2%} | "
                        f"{metrics['f1']:.2%} |")
        lines.append("")
    
    # Analysis
    lines.append("## Analysis")
    lines.append("")
    
    if best_model:
        improvement = best_accuracy - baseline['accuracy']
        lines.append(f"The best performing model-based approach ({best_model}) achieves "
                    f"{best_accuracy:.2%} accuracy, which is ")
        
        if improvement > 0:
            lines.append(f"**{improvement:.2%} higher** than the Logical DNA baseline.")
        elif improvement < 0:
            lines.append(f"**{abs(improvement):.2%} lower** than the Logical DNA baseline.")
        else:
            lines.append("**equal** to the Logical DNA baseline.")
        
        lines.append("")
    
    # Success criteria
    lines.append("## Success Criteria Assessment")
    lines.append("")
    
    success_threshold = 0.15  # 15%
    if best_accuracy >= success_threshold:
        lines.append(f"✅ **Success**: Best model achieves {best_accuracy:.2%} accuracy, "
                    f"exceeding the {success_threshold:.0%} threshold (3× random baseline).")
    else:
        lines.append(f"❌ **Below Threshold**: Best model achieves {best_accuracy:.2%} accuracy, "
                    f"below the {success_threshold:.0%} threshold.")
    
    lines.append("")
    
    # Notes
    lines.append("## Notes")
    lines.append("")
    lines.append("- All results are from zero-shot embeddings (no fine-tuning)")
    lines.append("- Random baseline for 20-class classification: 5%")
    lines.append("- Comparison is directional only, not statistically validated")
    lines.append("- CLAVE may not be available if model download failed")
    lines.append("")
    
    report_content = "\n".join(lines)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Report saved to {output_path}")
    
    return report_content


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison report for model-based authorship verification"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the report (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Generate report
    report = generate_report(args.output)
    
    # Also print to stdout
    print("\n" + "="*70)
    print(report)
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
