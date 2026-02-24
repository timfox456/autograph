# Per-Class Performance Report

**Date**: 2026-02-23  
**Project**: Autograph-DS Model Evaluation  
**Dataset**: dataset_329_features.csv (946 samples, 329 features)  
**Test Set**: 190 samples (20% hold-out)

---

## Executive Summary

This report analyzes model performance across the two main classes: **AI-generated code** and **Human-written code**. The models achieve exceptional performance with >98% accuracy on both classes.

---

## Class Distribution

| Class | Training Samples | Test Samples | Percentage |
|-------|------------------|--------------|------------|
| **AI** | 212 | 53 | 27.9% |
| **Human** | 548 | 137 | 72.1% |
| **Total** | 760 | 190 | 100% |

**Note**: Dataset is imbalanced with more human samples (72%) than AI samples (28%).

---

## RandomForest Performance by Class

### Per-Class Metrics

| Class | Precision | Recall | F1 Score | Support | Error Rate |
|-------|-----------|--------|----------|---------|------------|
| **AI** | 0.9464 | 1.0000 | 0.9725 | 53 | 0.0% |
| **Human** | 1.0000 | 0.9781 | 0.9889 | 137 | 2.2% |
| **Macro Average** | 0.9732 | 0.9891 | 0.9807 | - | - |
| **Weighted Average** | 0.9851 | 0.9842 | 0.9843 | - | - |

### Confusion Matrix

| Actual \ Predicted | AI | Human |
|-------------------|-----|-------|
| **AI** | 53 | 0 |
| **Human** | 3 | 134 |

**Analysis**:
- **True Positives (AI)**: 53/53 = 100% recall
- **True Positives (Human)**: 134/137 = 97.8% recall
- **False Positives**: 3 human samples misclassified as AI
- **Overall Accuracy**: 98.4%

### Key Findings

1. **Perfect AI Detection**: RandomForest correctly identifies all 53 AI-generated samples (100% recall)
2. **Strong Human Detection**: 97.8% recall for human samples
3. **Minor Confusion**: Only 3 human samples misclassified as AI (2.2% error rate)
4. **Precision**: Perfect precision for human class (1.0), very high for AI (0.946)

---

## XGBoost Performance by Class

### Per-Class Metrics

| Class | Precision | Recall | F1 Score | Support | Error Rate |
|-------|-----------|--------|----------|---------|------------|
| **AI** | 0.9636 | 1.0000 | 0.9815 | 53 | 0.0% |
| **Human** | 1.0000 | 0.9854 | 0.9926 | 137 | 1.5% |
| **Macro Average** | 0.9818 | 0.9927 | 0.9871 | - | - |
| **Weighted Average** | 0.9899 | 0.9895 | 0.9895 | - | - |

### Confusion Matrix

| Actual \ Predicted | AI | Human |
|-------------------|-----|-------|
| **AI** | 53 | 0 |
| **Human** | 2 | 135 |

**Analysis**:
- **True Positives (AI)**: 53/53 = 100% recall
- **True Positives (Human)**: 135/137 = 98.5% recall
- **False Positives**: 2 human samples misclassified as AI
- **Overall Accuracy**: 98.9%

### Key Findings

1. **Perfect AI Detection**: XGBoost also achieves 100% recall on AI samples
2. **Better Human Detection**: 98.5% recall (vs 97.8% for RF)
3. **Fewer Errors**: Only 2 misclassifications (vs 3 for RF)
4. **Higher Precision**: 0.964 for AI class (vs 0.946 for RF)

---

## Model Comparison by Class

### AI Class Performance

| Metric | RandomForest | XGBoost | Winner | Difference |
|--------|--------------|---------|--------|------------|
| Precision | 0.9464 | 0.9636 | XGBoost | +1.72% |
| Recall | 1.0000 | 1.0000 | Tie | 0.00% |
| F1 Score | 0.9725 | 0.9815 | XGBoost | +0.90% |

### Human Class Performance

| Metric | RandomForest | XGBoost | Winner | Difference |
|--------|--------------|---------|--------|------------|
| Precision | 1.0000 | 1.0000 | Tie | 0.00% |
| Recall | 0.9781 | 0.9854 | XGBoost | +0.73% |
| F1 Score | 0.9889 | 0.9926 | XGBoost | +0.37% |

### Overall Comparison

| Metric | RandomForest | XGBoost | Improvement |
|--------|--------------|---------|-------------|
| Accuracy | 98.42% | 98.95% | +0.53% |
| F1 (macro) | 98.07% | 98.71% | +0.64% |
| ROC-AUC | 99.57% | 99.92% | +0.35% |
| Misclassifications | 3 | 2 | -33% |

---

## Error Analysis

### Misclassified Samples

**RandomForest (3 errors)**:
- 3 human samples classified as AI
- Error rate: 2.2% of human samples

**XGBoost (2 errors)**:
- 2 human samples classified as AI
- Error rate: 1.5% of human samples

**Pattern**: Both models occasionally misclassify human code as AI, but never the reverse. This suggests:
1. Some human-written code has AI-like characteristics
2. Models are conservative in labeling AI (high precision for AI class)
3. No false negatives for AI detection (perfect recall)

---

## Key Findings

### 1. Exceptional Performance on Both Classes
- Both models achieve >98% accuracy
- Perfect recall for AI detection (no missed AI samples)
- Very high precision for human detection (no false human positives)

### 2. XGBoost Superior Across All Metrics
- Better precision on AI class (+1.72%)
- Better recall on human class (+0.73%)
- Fewer total errors (2 vs 3)
- Higher ROC-AUC (99.92% vs 99.57%)

### 3. Class Imbalance Handled Well
- Despite 72% human / 28% AI split
- Both classes achieve >97% F1 scores
- No bias toward majority class

### 4. Conservative AI Detection
- Models prefer labeling as AI when uncertain
- Results in high AI recall (100%) but slightly lower precision
- No AI samples slip through undetected

---

## Recommendations

### Immediate Actions

1. ✅ **Deploy XGBoost as Primary Model**
   - Superior performance on both classes
   - Fewer misclassifications
   - Higher confidence scores

2. ✅ **Use RandomForest as Backup**
   - Nearly identical performance
   - Different algorithm provides diversity
   - Useful for ensemble methods

3. ✅ **Monitor Human→AI Misclassifications**
   - Investigate the 2-3 human samples misclassified as AI
   - May reveal edge cases or labeling errors
   - Could improve training data quality

### Future Improvements

1. **Collect More AI Samples**
   - Current ratio: 28% AI / 72% Human
   - Target ratio: 40% AI / 60% Human
   - Would improve AI class precision

2. **Analyze Misclassified Samples**
   - Manual review of the 2-3 errors
   - Identify common patterns
   - Add specialized features if needed

3. **Confidence Threshold Tuning**
   - Current: Default 0.5 threshold
   - Consider: 0.7 threshold for higher precision
   - Trade-off: May reduce recall slightly

---

## Conclusion

Both RandomForest and XGBoost demonstrate **exceptional performance** on the AI vs Human classification task:

- **Perfect AI Detection**: 100% recall, no missed AI samples
- **Strong Human Detection**: >97% recall, minimal false positives
- **Overall Accuracy**: >98% on both models
- **Clear Winner**: XGBoost with 98.95% accuracy and only 2 errors

The models successfully distinguish between AI-generated and human-written code with near-perfect accuracy, making them suitable for production deployment in the Autograph-DS attestation engine.

---

## Appendix: Raw Metrics

### RandomForest
```json
{
  "ai": {
    "precision": 0.9464285714285714,
    "recall": 1.0,
    "f1": 0.9724770642201835,
    "support": 53
  },
  "human": {
    "precision": 1.0,
    "recall": 0.9781021897810219,
    "f1": 0.988929889298893,
    "support": 137
  }
}
```

### XGBoost
```json
{
  "ai": {
    "precision": 0.9636363636363636,
    "recall": 1.0,
    "f1": 0.9814814814814815,
    "support": 53
  },
  "human": {
    "precision": 1.0,
    "recall": 0.9854014598540146,
    "f1": 0.9926470588235294,
    "support": 137
  }
}
```

---

*Report generated by Autograph-DS Model Evaluation Pipeline*
