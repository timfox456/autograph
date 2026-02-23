# Feature Importance Analysis — Autograph Logical DNA

> **Experiment date**: 2026-02-23
> **Script**: `report_feature_importance.py --permutation`
> **Model**: Random Forest, 1 000 trees, `class_weight='balanced'`, OOB enabled

---

## 1. Objective

Determine which Logical DNA features contribute most to distinguishing code
authorship identities. Two complementary methods are used:

| Method | What it measures | Bias |
|---|---|---|
| **MDI** (Mean Decrease Impurity) | How much each feature reduces Gini impurity across all tree splits | Favours high-cardinality and continuous features |
| **Permutation Importance** | How much test-set accuracy drops when a feature's values are shuffled | More robust; measures actual predictive contribution |

Both are reported so their agreement (and disagreement) can inform feature
selection and privacy trade-off decisions.

---

## 2. Dataset

| Property | Value |
|---|---|
| Samples | 753 |
| Features | 566 |
| Identities | 19 |
| Human samples | 681 (90.4%) |
| AI samples | 72 (9.6%) |

### Identity Distribution

| Identity | Samples | Type |
|---|---|---|
| mariusz | 82 | Human |
| samuel | 50 | Human |
| ian | 50 | Human |
| sebastian | 50 | Human |
| armin | 50 | Human |
| david | 50 | Human |
| donald | 50 | Human |
| will | 50 | Human |
| glyph | 50 | Human |
| tom | 50 | Human |
| lukasz | 50 | Human |
| audrey | 44 | Human |
| alex | 31 | Human |
| hynek | 24 | Human |
| kimi | 15 | AI |
| gpt4o | 15 | AI |
| gemini | 15 | AI |
| deepseek_v3 | 14 | AI |
| claude | 13 | AI |

The dataset is imbalanced: human authors dominate, and AI identities have the
fewest samples (13–15 each). The RF uses `class_weight='balanced'` to compensate.

---

## 3. Model Performance

| Metric | Value |
|---|---|
| Trees | 1 000 |
| OOB Score | 0.7176 |
| Train Accuracy | 0.9983 |
| Test Accuracy | 0.6887 |
| Macro-avg F1 | 0.59 |
| Weighted-avg F1 | 0.68 |
| Training Time | 0.8 s |

The large train–test gap (~0.31) indicates overfitting, expected with 566 features
on only 753 samples. The OOB score (0.72) is a more honest estimate than train
accuracy and closely tracks test accuracy (0.69).

### Per-Identity Performance (Test Set)

| Identity | Precision | Recall | F1 | Support | Notes |
|---|---|---|---|---|---|
| david | 1.00 | 1.00 | 1.00 | 10 | Perfect — strong stylistic signal |
| sebastian | 0.91 | 1.00 | 0.95 | 10 | |
| tom | 0.91 | 1.00 | 0.95 | 10 | |
| audrey | 0.80 | 0.89 | 0.84 | 9 | |
| will | 0.88 | 0.70 | 0.78 | 10 | |
| hynek | 1.00 | 0.60 | 0.75 | 5 | High precision, low recall |
| ian | 0.86 | 0.60 | 0.71 | 10 | |
| armin | 0.70 | 0.70 | 0.70 | 10 | |
| glyph | 0.62 | 0.80 | 0.70 | 10 | |
| alex | 1.00 | 0.50 | 0.67 | 6 | |
| samuel | 0.75 | 0.60 | 0.67 | 10 | |
| lukasz | 0.83 | 0.50 | 0.62 | 10 | |
| mariusz | 0.44 | 0.94 | 0.60 | 16 | Low precision — catches all but over-predicts |
| donald | 0.80 | 0.40 | 0.53 | 10 | |
| claude | 0.33 | 0.33 | 0.33 | 3 | Too few samples |
| gpt4o | 0.33 | 0.33 | 0.33 | 3 | Too few samples |
| deepseek_v3 | 0.00 | 0.00 | 0.00 | 3 | Unlearnable at this scale |
| gemini | 0.00 | 0.00 | 0.00 | 3 | Unlearnable at this scale |
| kimi | 0.00 | 0.00 | 0.00 | 3 | Unlearnable at this scale |

**Observation**: Human identities with 50 samples perform well (F1 0.53–1.00).
AI identities with 13–15 samples struggle badly — they receive only 3 test
samples each, making evaluation noisy and learning difficult.

---

## 4. MDI Feature Importance (Mean Decrease Impurity)

### Top 40 Features

| Rank | Feature | Importance | Pct | Cumulative | Bucket |
|---:|---|---:|---:|---:|---|
| 1 | `avg_vertical_chunk_size` | 0.01538 | 1.54% | 1.5% | layout_rhythm |
| 2 | `blank_line_ratio` | 0.01435 | 1.44% | 3.0% | layout_rhythm |
| 3 | `max_consecutive_newlines` | 0.01183 | 1.18% | 4.2% | layout_rhythm |
| 4 | `quote_single` | 0.01077 | 1.08% | 5.2% | micro_stylistics |
| 5 | `quote_double` | 0.01067 | 1.07% | 6.3% | micro_stylistics |
| 6 | `identifier_entropy` | 0.01021 | 1.02% | 7.3% | lexical_complexity |
| 7 | `snake_case_ratio` | 0.00978 | 0.98% | 8.3% | logical_idioms |
| 8 | `avg_identifier_length` | 0.00908 | 0.91% | 9.2% | lexical_complexity |
| 9 | `exit_density` | 0.00820 | 0.82% | 10.0% | cfg_complexity |
| 10 | `node_identifier` | 0.00813 | 0.81% | 10.8% | structural_topology |
| 11 | `camel_case_words_ratio` | 0.00802 | 0.80% | 11.6% | string_content |
| 12 | `node_block` | 0.00791 | 0.79% | 12.4% | structural_topology |
| 13 | `node_return_statement` | 0.00785 | 0.78% | 13.2% | structural_topology |
| 14 | `node_expression_statement` | 0.00770 | 0.77% | 14.0% | structural_topology |
| 15 | `avg_branching_factor` | 0.00767 | 0.77% | 14.8% | structural_topology |
| 16 | `node_.` | 0.00766 | 0.77% | 15.5% | structural_topology |
| 17 | `short_identifier_ratio` | 0.00763 | 0.76% | 16.3% | lexical_complexity |
| 18 | `node_return` | 0.00748 | 0.75% | 17.0% | structural_topology |
| 19 | `node_(` | 0.00741 | 0.74% | 17.8% | structural_topology |
| 20 | `node_)` | 0.00739 | 0.74% | 18.5% | structural_topology |
| 21 | `node_attribute` | 0.00725 | 0.73% | 19.2% | structural_topology |
| 22 | `decisions_per_line` | 0.00723 | 0.72% | 20.0% | cyclomatic_complexity |
| 23 | `node_,` | 0.00718 | 0.72% | 20.7% | structural_topology |
| 24 | `top_trigrams_string_start:string_content:string_end` | 0.00685 | 0.68% | 21.4% | ast_trigrams |
| 25 | `node_argument_list` | 0.00675 | 0.67% | 22.0% | structural_topology |
| 26 | `node_module` | 0.00655 | 0.66% | 22.7% | structural_topology |
| 27 | `node_string_start` | 0.00652 | 0.65% | 23.3% | structural_topology |
| 28 | `node_call` | 0.00651 | 0.65% | 24.0% | structural_topology |
| 29 | `total_nodes` | 0.00628 | 0.63% | 24.6% | structural_topology |
| 30 | `node_:` | 0.00627 | 0.63% | 25.2% | structural_topology |
| 31 | `node_=` | 0.00627 | 0.63% | 25.9% | structural_topology |
| 32 | `node_string_content` | 0.00616 | 0.62% | 26.5% | structural_topology |
| 33 | `total_string_literals` | 0.00613 | 0.61% | 27.1% | string_content |
| 34 | `method_call_ratio` | 0.00597 | 0.60% | 27.7% | semantic_fingerprint |
| 35 | `top_trigrams_string:string_start:string_content` | 0.00594 | 0.59% | 28.3% | ast_trigrams |
| 36 | `camel_case_ratio` | 0.00590 | 0.59% | 28.9% | logical_idioms |
| 37 | `node_string` | 0.00589 | 0.59% | 29.5% | structural_topology |
| 38 | `node_integer` | 0.00572 | 0.57% | 30.0% | structural_topology |
| 39 | `uses_fastapi` | 0.00570 | 0.57% | 30.6% | semantic_fingerprint |
| 40 | `node_assignment` | 0.00570 | 0.57% | 31.2% | structural_topology |

**Key takeaway**: The top 10 features account for only 10.8% of total importance.
No single feature dominates — the RF spreads importance broadly, which is typical
for large forests and high-dimensional feature spaces.

---

## 5. Importance by Feature Bucket

### Aggregate MDI Importance

| Bucket | Privacy | Features | Total | Share | Mean | Max |
|---|---|---:|---:|---:|---:|---:|
| **structural_topology** | High | 123 | 0.36617 | **36.62%** | 0.002977 | 0.00813 |
| **ast_trigrams** | High | 150 | 0.29381 | **29.38%** | 0.001959 | 0.00685 |
| semantic_fingerprint | Medium | 165 | 0.09716 | 9.72% | 0.000589 | 0.00597 |
| string_content | Low | 58 | 0.05754 | 5.75% | 0.000992 | 0.00802 |
| layout_rhythm | Medium | 3 | 0.04156 | 4.16% | 0.013852 | 0.01538 |
| cyclomatic_complexity | Medium | 31 | 0.03928 | 3.93% | 0.001267 | 0.00723 |
| micro_stylistics | Medium | 7 | 0.03117 | 3.12% | 0.004453 | 0.01077 |
| lexical_complexity | Low | 3 | 0.02691 | 2.69% | 0.008971 | 0.01021 |
| logical_idioms | Low | 6 | 0.01962 | 1.96% | 0.003269 | 0.00978 |
| comment_stylistics | Low | 12 | 0.01255 | 1.25% | 0.001046 | 0.00394 |
| cfg_complexity | Medium | 4 | 0.01033 | 1.03% | 0.002583 | 0.00820 |
| logic_flow | High | 1 | 0.00288 | 0.29% | 0.002879 | 0.00288 |
| syntactic_bias | Medium | 3 | 0.00103 | 0.10% | 0.000342 | 0.00055 |

### Privacy-Level Aggregation

| Privacy Level | Buckets | Features | Combined Share |
|---|---|---:|---:|
| **High** (safe for public attestation) | structural_topology, ast_trigrams, logic_flow | 274 | **66.3%** |
| **Medium** | semantic_fingerprint, layout_rhythm, cyclomatic_complexity, micro_stylistics, cfg_complexity, syntactic_bias | 213 | **22.1%** |
| **Low** (reveals coding style) | string_content, lexical_complexity, logical_idioms, comment_stylistics | 79 | **11.7%** |

**This is the most actionable finding**: Two-thirds of the model's discriminative
power comes from high-privacy features (structural topology + AST trigrams) that
do not reveal identifiable coding style. This validates Autograph's
privacy-preserving attestation design.

---

## 6. Cumulative Importance Concentration

| Threshold | Features Required | % of Total (566) |
|---:|---:|---:|
| 50% | 82 | 14.5% |
| 75% | 175 | 30.9% |
| 80% | 202 | 35.7% |
| 90% | 273 | 48.2% |
| 95% | 324 | 57.2% |
| 99% | 399 | 70.5% |

Half the total importance is concentrated in 82 features (14.5% of the feature
space). The remaining 242 features (43%) contribute less than 5% combined.
This suggests significant dimensionality reduction is possible.

### Near-Zero Importance Features

**40 features (7.1%)** have near-zero importance (< 1e-6):

| Bucket | Near-Zero | Total | Pct |
|---|---:|---:|---:|
| semantic_fingerprint | 21 | 165 | 12.7% |
| string_content | 12 | 58 | 20.7% |
| cyclomatic_complexity | 6 | 31 | 19.4% |
| comment_stylistics | 1 | 12 | 8.3% |

The semantic_fingerprint bucket has the most dead weight — many `call_freq_*` and
`uses_*` columns are too sparse to be useful at the current dataset scale.

---

## 7. Permutation Importance

Permutation importance measures the actual impact on test accuracy when a feature
is shuffled, avoiding MDI's known bias toward continuous features.

### Top 20 Features (Permutation)

| Rank | Feature | Importance | ±Std | Bucket |
|---:|---|---:|---:|---|
| 1 | `quote_single` | 0.04768 | 0.01351 | micro_stylistics |
| 2 | `quote_doubledoubledouble` | 0.02649 | 0.00662 | micro_stylistics |
| 3 | `top_trigrams_.:identifier:[` | 0.02053 | 0.00357 | ast_trigrams |
| 4 | `is_sphinx_style` | 0.01325 | 0.00000 | string_content |
| 5 | `top_trigrams_identifier:[:identifier` | 0.01258 | 0.00199 | ast_trigrams |
| 6 | `top_trigrams_(:)::` | 0.01192 | 0.00265 | ast_trigrams |
| 7 | `min_cyclomatic_complexity` | 0.01192 | 0.00265 | cyclomatic_complexity |
| 8 | `node_\|` | 0.01192 | 0.00265 | structural_topology |
| 9 | `top_trigrams_ERROR:identifier:identifier` | 0.01192 | 0.00265 | ast_trigrams |
| 10 | `has_docstrings` | 0.01192 | 0.00265 | comment_stylistics |
| 11 | `node_not` | 0.01126 | 0.00424 | structural_topology |
| 12 | `node_is` | 0.01126 | 0.00424 | structural_topology |
| 13 | `top_trigrams_subscript:attribute:identifier` | 0.01060 | 0.00324 | ast_trigrams |
| 14 | `node_identifier` | 0.00993 | 0.00444 | structural_topology |
| 15 | `camel_case_words_ratio` | 0.00993 | 0.00611 | string_content |
| 16 | `snake_case_ratio` | 0.00927 | 0.00530 | logical_idioms |
| 17 | `node_subscript` | 0.00927 | 0.00324 | structural_topology |
| 18 | `avg_cyclomatic_complexity` | 0.00861 | 0.00424 | cyclomatic_complexity |
| 19 | `top_trigrams_attribute:identifier:.` | 0.00795 | 0.00649 | ast_trigrams |
| 20 | `top_trigrams_expression_statement:string:string_start` | 0.00795 | 0.00496 | ast_trigrams |

---

## 8. MDI vs Permutation — Agreement Analysis

### Top-20 Overlap

Only **4 of 20** features appear in both top-20 lists:

| Feature | MDI Rank | Perm Rank | Bucket |
|---|---:|---:|---|
| `node_identifier` | 10 | 14 | structural_topology |
| `camel_case_words_ratio` | 11 | 15 | string_content |
| `snake_case_ratio` | 7 | 16 | logical_idioms |
| `quote_single` | 4 | 1 | micro_stylistics |

### Features Unique to Each Method

**MDI-only** (top 20): Dominated by layout/rhythm and generic AST node counts —
continuous features with high cardinality that get many splits but don't
individually change predictions much:
- `avg_vertical_chunk_size`, `blank_line_ratio`, `max_consecutive_newlines`
- `identifier_entropy`, `avg_identifier_length`, `short_identifier_ratio`
- `node_block`, `node_return_statement`, `node_expression_statement`
- `node_.`, `node_(`, `node_)`, `node_return`, `avg_branching_factor`
- `exit_density`, `quote_double`

**Permutation-only** (top 20): Dominated by specific AST trigrams and rare but
decisive signals — these features have sparse or categorical distributions but
strongly separate specific identities:
- `top_trigrams_.:identifier:[` — subscript access pattern
- `top_trigrams_(:)::` — empty-parens-then-colon pattern
- `top_trigrams_ERROR:identifier:identifier` — parse-error-adjacent patterns
- `top_trigrams_identifier:[:identifier` — dict/index access style
- `top_trigrams_subscript:attribute:identifier`
- `is_sphinx_style` — Sphinx docstring convention
- `has_docstrings`
- `node_|` — union type annotations (Python 3.10+)
- `node_not`, `node_is` — boolean operator preference
- `min_cyclomatic_complexity`, `avg_cyclomatic_complexity`

### Interpretation

The low overlap (4/20) is expected and informative:

1. **MDI** tells us what the forest *uses often*. High-cardinality continuous
   features provide many splitting opportunities — they appear important by
   construction, even when their marginal predictive contribution is modest.

2. **Permutation** tells us what *actually matters for accuracy*. Features like
   `is_sphinx_style` or `node_|` may only split a few samples, but those splits
   are decisive for distinguishing specific identities.

**For feature selection**, permutation importance should be preferred. For
understanding the model's internal structure, MDI is useful.

---

## 9. Top 5 Features per Bucket

### layout_rhythm (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 1 | `avg_vertical_chunk_size` | 0.01538 |
| 2 | `blank_line_ratio` | 0.01435 |
| 3 | `max_consecutive_newlines` | 0.01183 |

All 3 features in this bucket rank in the global top 3 by MDI. Vertical spacing
habits are deeply personal — some developers use liberal blank lines, others keep
code dense. This bucket has the **highest mean importance** (0.0139) despite
having only 3 features.

### micro_stylistics (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 4 | `quote_single` | 0.01077 |
| 5 | `quote_double` | 0.01067 |
| 60 | `quote_doubledoubledouble` | 0.00435 |
| 74 | `indent_width` | 0.00410 |
| 278 | `trailing_commas_count` | 0.00112 |

Quote preference is one of the strongest individual signals. `quote_single` ranks
#1 in permutation importance (0.048), meaning shuffling it drops accuracy by ~5%.
This makes intuitive sense: `'string'` vs `"string"` is a near-binary habit.

### lexical_complexity (Privacy: Low)

| Rank | Feature | MDI |
|---:|---|---:|
| 6 | `identifier_entropy` | 0.01021 |
| 8 | `avg_identifier_length` | 0.00908 |
| 17 | `short_identifier_ratio` | 0.00763 |

Naming diversity and verbosity are strong author signals. Some developers prefer
terse names (`x`, `df`, `n`), others use descriptive names
(`processed_dataframe`, `iteration_count`).

### logical_idioms (Privacy: Low)

| Rank | Feature | MDI |
|---:|---|---:|
| 7 | `snake_case_ratio` | 0.00978 |
| 36 | `camel_case_ratio` | 0.00590 |
| 185 | `class_definition_count` | 0.00190 |
| 290 | `f_string_ratio` | 0.00104 |
| 368 | `list_comprehension_count` | 0.00051 |

Naming convention (snake_case vs camelCase) is the strongest idiomatic signal.
f-string adoption and comprehension usage are weaker — possibly because most
modern Python code converges on f-strings.

### structural_topology (Privacy: High)

| Rank | Feature | MDI |
|---:|---|---:|
| 10 | `node_identifier` | 0.00813 |
| 12 | `node_block` | 0.00791 |
| 13 | `node_return_statement` | 0.00785 |
| 14 | `node_expression_statement` | 0.00770 |
| 15 | `avg_branching_factor` | 0.00767 |

The AST node distribution forms a structural fingerprint. The ratio of identifiers
to blocks to returns differs by coding style: functional code has more returns,
imperative code has more assignments and expression statements.

### ast_trigrams (Privacy: High)

| Rank | Feature | MDI |
|---:|---|---:|
| 24 | `string_start:string_content:string_end` | 0.00685 |
| 35 | `string:string_start:string_content` | 0.00594 |
| 42 | `attribute:identifier:.` | 0.00565 |
| 46 | `identifier:.:identifier` | 0.00532 |
| 56 | `expression_statement:string:string_start` | 0.00457 |

String-related trigrams dominate — reflecting how authors handle string literals.
The `attribute:identifier:.` trigram (method chaining) is notable: chaining style
varies significantly between developers.

### semantic_fingerprint (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 34 | `method_call_ratio` | 0.00597 |
| 39 | `uses_fastapi` | 0.00570 |
| 41 | `keyword_arg_ratio` | 0.00567 |
| 67 | `print_calls_ratio` | 0.00425 |
| 69 | `call_freq_print` | 0.00416 |

`uses_fastapi` is a strong signal because it's likely specific to one or two
identities in the corpus. This is a dataset artefact rather than a generalizable
feature — more diverse data would dilute single-library signals.

### cyclomatic_complexity (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 22 | `decisions_per_line` | 0.00723 |
| 57 | `total_decision_points` | 0.00441 |
| 79 | `avg_cyclomatic_complexity` | 0.00383 |
| 85 | `max_cyclomatic_complexity` | 0.00356 |
| 87 | `total_functions_analyzed` | 0.00351 |

Code complexity profiles differ by authorship — some prefer simple functions,
others write complex control flow. `decisions_per_line` (rank 22) is the strongest
signal in this bucket.

### comment_stylistics (Privacy: Low)

| Rank | Feature | MDI |
|---:|---|---:|
| 76 | `comment_to_code_ratio` | 0.00394 |
| 159 | `length_variance` | 0.00226 |
| 178 | `sentence_case_ratio` | 0.00200 |
| 232 | `dead_code_density` | 0.00141 |
| 252 | `has_docstrings` | 0.00127 |

Comment patterns are surprisingly weak in MDI but `has_docstrings` ranks #10 in
permutation importance. This suggests docstring presence is a decisive binary
signal for separating specific identities (e.g., well-documented vs. no-docs).

### cfg_complexity (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 9 | `exit_density` | 0.00820 |
| 174 | `guard_clause_score` | 0.00204 |
| 471 | `break_statement_count` | 0.00005 |
| 476 | `while_true_ratio` | 0.00004 |

`exit_density` (rank 9) is the standout — how frequently functions use early
returns, raises, or continues. Guard clause style and loop patterns contribute
almost nothing.

### syntactic_bias (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 362 | `literal_collection_ratio` | 0.00055 |
| 377 | `exception_depth` | 0.00045 |
| 493 | `boolean_style_score` | 0.00002 |

This is the weakest bucket overall (0.10% of total). These features likely need
more diverse data or refinement to become useful.

### logic_flow (Privacy: High)

| Rank | Feature | MDI |
|---:|---|---:|
| 115 | `functional_score` | 0.00288 |

A single feature measuring functional vs procedural bias. Moderate importance.

---

## 10. Actionable Recommendations

### Feature Selection

1. **Safe to prune**: The bottom ~170 features (30%) contribute < 5% total
   importance. Removing them would simplify the model with negligible accuracy loss.

2. **Candidate compact model**: A 82-feature model captures 50% of importance.
   Combined with permutation-guided selection, a 60–100 feature model could match
   or exceed the full model's generalization (less overfitting).

3. **Dead features to remove**: 40 near-zero features (mostly sparse `call_freq_*`
   and `exception_type_*` columns) are noise and should be dropped.

### Privacy Trade-offs

4. **High-privacy attestation is viable**: Structural topology + AST trigrams alone
   provide 66% of discriminative signal. A privacy-preserving mode using only
   High-privacy buckets should retain substantial accuracy.

5. **Layout rhythm is a privacy bargain**: 3 features, Medium privacy, 4.2% of
   importance. Adding these to the High-privacy set would boost signal cheaply.

### Data Collection

6. **AI identities need more samples**: 13–15 samples per AI model is insufficient.
   The model cannot learn these identities reliably. Target ≥ 50 samples per identity.

7. **Library-specific features (`uses_fastapi`, etc.) are dataset artefacts**: With
   more diverse data, these will naturally diminish. Don't over-index on them.

### Model Improvements

8. **Try a reduced-feature RF or XGBoost**: Train on the top 80–100 features by
   permutation importance. This should reduce overfitting (current gap: 0.31
   between train and test accuracy).

9. **Cross-validate**: The current single 80/20 split with only 3 test samples for
   AI identities is unreliable. Use stratified k-fold (k=5) for robust estimates.

---

## Appendix: Reproducing This Analysis

```bash
cd autograph-ds

# MDI only (fast, ~1s)
python report_feature_importance.py

# MDI + Permutation importance (~2 min)
python report_feature_importance.py --permutation

# Customize
python report_feature_importance.py --n-trees 2000 --top-n 50 --perm-repeats 20
```
