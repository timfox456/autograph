# Feature Importance Analysis — Autograph Logical DNA

> **Experiment date**: 2026-02-23
> **Script**: `report_feature_importance.py --permutation`
> **Model**: Random Forest, 1 000 trees, `class_weight='balanced'`, OOB enabled
> **Previous run**: 2026-02-23 (pre–AI sample expansion, 753 samples, 72 AI)

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

### What Changed Since Last Run

The prior analysis suffered from **sparse AI attestation**: only 13–15 samples
per AI identity (72 total), making AI classes unlearnable. We expanded the AI
sample collection to 50–55 samples per model (265 total), bringing AI identities
to parity with human authors. Key impacts:

- **Dataset**: 753 → 946 samples (+26%), AI share 9.6% → 28.0%
- **Test accuracy**: 0.689 → 0.721 (+3.2 pp)
- **Macro F1**: 0.59 → 0.72 (+13 pp) — driven almost entirely by AI identities
  becoming learnable
- **AI F1 scores**: gemini 0.00→0.70, claude 0.33→0.64, gpt4o 0.33→0.64
- **Permutation importance** shifted significantly — layout rhythm and
  semantic features (print usage, import patterns) gained prominence as
  human-vs-AI discriminators

---

## 2. Dataset

| Property | Value | Previous |
|---|---|---|
| Samples | 946 | 753 |
| Features | 572 | 566 |
| Identities | 19 | 19 |
| Human samples | 681 (72.0%) | 681 (90.4%) |
| AI samples | 265 (28.0%) | 72 (9.6%) |

### Identity Distribution

| Identity | Samples | Type | Δ from previous |
|---|---|---|---|
| mariusz | 82 | Human | — |
| gemini | 55 | AI | +40 |
| kimi | 54 | AI | +39 |
| gpt4o | 54 | AI | +39 |
| claude | 52 | AI | +39 |
| deepseek_v3 | 50 | AI | +36 |
| samuel | 50 | Human | — |
| glyph | 50 | Human | — |
| sebastian | 50 | Human | — |
| armin | 50 | Human | — |
| david | 50 | Human | — |
| donald | 50 | Human | — |
| will | 50 | Human | — |
| tom | 50 | Human | — |
| lukasz | 50 | Human | — |
| ian | 50 | Human | — |
| audrey | 44 | Human | — |
| alex | 31 | Human | — |
| hynek | 24 | Human | — |

AI identities now have 50–55 samples each (previously 13–15), reaching parity
with most human authors. The RF uses `class_weight='balanced'` to further
compensate for the remaining human-majority imbalance.

---

## 3. Model Performance

| Metric | Value | Previous | Δ |
|---|---|---|---|
| Trees | 1 000 | 1 000 | — |
| OOB Score | 0.6984 | 0.7176 | −0.019 |
| Train Accuracy | 0.9987 | 0.9983 | +0.000 |
| Test Accuracy | 0.7211 | 0.6887 | **+0.032** |
| Macro-avg F1 | 0.72 | 0.59 | **+0.13** |
| Weighted-avg F1 | 0.72 | 0.68 | +0.04 |
| Training Time | 0.8 s | 0.8 s | — |

The train–test gap narrowed from ~0.31 to ~0.28. The OOB score dipped slightly
(0.718 → 0.698) as the model must now learn 5 additional AI classes with
distinctive but overlapping patterns. However, test accuracy and macro F1 both
improved substantially — the model generalises better to unseen AI samples.

### Per-Identity Performance (Test Set)

| Identity | Precision | Recall | F1 | Support | Notes |
|---|---|---|---|---|---|
| david | 1.00 | 1.00 | 1.00 | 10 | Perfect — strong stylistic signal |
| tom | 1.00 | 1.00 | 1.00 | 10 | Perfect — previously 0.95 |
| sebastian | 1.00 | 0.90 | 0.95 | 10 | |
| audrey | 1.00 | 0.89 | 0.94 | 9 | Improved from 0.84 |
| will | 0.90 | 0.90 | 0.90 | 10 | Improved from 0.78 |
| ian | 0.89 | 0.80 | 0.84 | 10 | Improved from 0.71 |
| glyph | 0.71 | 1.00 | 0.83 | 10 | Improved from 0.70 |
| samuel | 1.00 | 0.60 | 0.75 | 10 | |
| gemini | 0.67 | 0.73 | 0.70 | 11 | **Was 0.00** — now learnable |
| armin | 0.64 | 0.70 | 0.67 | 10 | |
| alex | 1.00 | 0.50 | 0.67 | 6 | |
| donald | 0.75 | 0.60 | 0.67 | 10 | Improved from 0.53 |
| lukasz | 0.64 | 0.70 | 0.67 | 10 | Improved from 0.62 |
| claude | 0.58 | 0.70 | 0.64 | 10 | **Was 0.33** — much improved |
| gpt4o | 0.64 | 0.64 | 0.64 | 11 | **Was 0.33** — much improved |
| mariusz | 0.52 | 0.76 | 0.62 | 17 | Still over-predicts |
| hynek | 0.67 | 0.40 | 0.50 | 5 | Declined from 0.75 — still too few samples |
| kimi | 0.56 | 0.45 | 0.50 | 11 | **Was 0.00** — now learnable |
| deepseek_v3 | 0.22 | 0.20 | 0.21 | 10 | **Was 0.00** — partially learnable |

**Key observation**: The sparse attestation fix worked. All five AI identities
are now at least partially learnable (F1 0.21–0.70), up from three being
completely unclassifiable. Claude, GPT-4o, and Gemini show solid performance
(F1 0.64–0.70). DeepSeek V3 remains the weakest — its coding style may
genuinely overlap with other AI models. Kimi shows moderate separability.

Human identities are largely stable or improved. The slight drop for hynek
(0.75 → 0.50) is expected with only 5 test samples — a single misclassification
swings F1 by 0.25.

---

## 4. MDI Feature Importance (Mean Decrease Impurity)

### Top 40 Features

| Rank | Feature | Importance | Pct | Cumulative | Bucket |
|---:|---|---:|---:|---:|---|
| 1 | `avg_vertical_chunk_size` | 0.01535 | 1.53% | 1.5% | layout_rhythm |
| 2 | `max_consecutive_newlines` | 0.01421 | 1.42% | 3.0% | layout_rhythm |
| 3 | `blank_line_ratio` | 0.01341 | 1.34% | 4.3% | layout_rhythm |
| 4 | `quote_double` | 0.01012 | 1.01% | 5.3% | micro_stylistics |
| 5 | `snake_case_ratio` | 0.00887 | 0.89% | 6.2% | logical_idioms |
| 6 | `identifier_entropy` | 0.00861 | 0.86% | 7.1% | lexical_complexity |
| 7 | `quote_single` | 0.00856 | 0.86% | 7.9% | micro_stylistics |
| 8 | `avg_branching_factor` | 0.00825 | 0.82% | 8.7% | structural_topology |
| 9 | `node_identifier` | 0.00766 | 0.77% | 9.5% | structural_topology |
| 10 | `avg_identifier_length` | 0.00764 | 0.76% | 10.3% | lexical_complexity |
| 11 | `camel_case_words_ratio` | 0.00750 | 0.75% | 11.0% | string_content |
| 12 | `node_.` | 0.00737 | 0.74% | 11.8% | structural_topology |
| 13 | `short_identifier_ratio` | 0.00714 | 0.71% | 12.5% | lexical_complexity |
| 14 | `node_attribute` | 0.00711 | 0.71% | 13.2% | structural_topology |
| 15 | `total_nodes` | 0.00701 | 0.70% | 13.9% | structural_topology |
| 16 | `node_,` | 0.00690 | 0.69% | 14.6% | structural_topology |
| 17 | `node_call` | 0.00685 | 0.68% | 15.3% | structural_topology |
| 18 | `node_module` | 0.00675 | 0.68% | 15.9% | structural_topology |
| 19 | `node_expression_statement` | 0.00663 | 0.66% | 16.6% | structural_topology |
| 20 | `camel_case_ratio` | 0.00657 | 0.66% | 17.3% | logical_idioms |
| 21 | `decisions_per_line` | 0.00639 | 0.64% | 17.9% | cyclomatic_complexity |
| 22 | `node_block` | 0.00623 | 0.62% | 18.5% | structural_topology |
| 23 | `node_)` | 0.00621 | 0.62% | 19.1% | structural_topology |
| 24 | `node_argument_list` | 0.00616 | 0.62% | 19.7% | structural_topology |
| 25 | `node_(` | 0.00615 | 0.62% | 20.4% | structural_topology |
| 26 | `node_:` | 0.00610 | 0.61% | 21.0% | structural_topology |
| 27 | `node_string` | 0.00607 | 0.61% | 21.6% | structural_topology |
| 28 | `node_string_start` | 0.00591 | 0.59% | 22.2% | structural_topology |
| 29 | `node_string_end` | 0.00587 | 0.59% | 22.8% | structural_topology |
| 30 | `quote_doubledoubledouble` | 0.00578 | 0.58% | 23.3% | micro_stylistics |
| 31 | `total_string_literals` | 0.00573 | 0.57% | 23.9% | string_content |
| 32 | `node_=` | 0.00569 | 0.57% | 24.5% | structural_topology |
| 33 | `regex_pattern_ratio` | 0.00566 | 0.57% | 25.0% | string_content |
| 34 | `exit_density` | 0.00552 | 0.55% | 25.6% | cfg_complexity |
| 35 | `node_string_content` | 0.00545 | 0.55% | 26.1% | structural_topology |
| 36 | `call_freq_FastAPI` | 0.00538 | 0.54% | 26.7% | semantic_fingerprint |
| 37 | `uses_fastapi` | 0.00531 | 0.53% | 27.2% | semantic_fingerprint |
| 38 | `top_trigrams_string:string_start:string_content` | 0.00530 | 0.53% | 27.7% | ast_trigrams |
| 39 | `method_call_ratio` | 0.00530 | 0.53% | 28.3% | semantic_fingerprint |
| 40 | `top_trigrams_expression_statement:string:string_start` | 0.00529 | 0.53% | 28.8% | ast_trigrams |

**Key takeaway**: The top 10 features account for 10.3% of total importance
(vs 10.8% previously). The distribution remains diffuse — no single feature
dominates. Layout rhythm still claims the top 3 positions, and the overall
ranking is remarkably stable despite adding 193 new samples.

### Notable Rank Changes (vs previous)

| Feature | Old Rank | New Rank | Direction |
|---|---:|---:|---|
| `exit_density` | 9 | 34 | ↓ significant drop |
| `avg_branching_factor` | 15 | 8 | ↑ rose to top 10 |
| `regex_pattern_ratio` | — | 33 | ↑ newly relevant |
| `camel_case_ratio` | 36 | 20 | ↑ rose notably |
| `node_return_statement` | 13 | >40 | ↓ dropped out of top 40 |

---

## 5. Importance by Feature Bucket

### Aggregate MDI Importance

| Bucket | Privacy | Features | Total | Share | Mean | Max |
|---|---|---:|---:|---:|---:|---:|
| **structural_topology** | High | 123 | 0.35192 | **35.19%** | 0.002861 | 0.00825 |
| **ast_trigrams** | High | 150 | 0.31119 | **31.12%** | 0.002075 | 0.00530 |
| semantic_fingerprint | Medium | 165 | 0.09917 | 9.92% | 0.000601 | 0.00538 |
| string_content | Low | 64 | 0.05830 | 5.83% | 0.000911 | 0.00750 |
| layout_rhythm | Medium | 3 | 0.04297 | 4.30% | 0.014324 | 0.01535 |
| cyclomatic_complexity | Medium | 31 | 0.03672 | 3.67% | 0.001185 | 0.00639 |
| micro_stylistics | Medium | 7 | 0.03021 | 3.02% | 0.004316 | 0.01012 |
| lexical_complexity | Low | 3 | 0.02339 | 2.34% | 0.007796 | 0.00861 |
| logical_idioms | Low | 6 | 0.02066 | 2.07% | 0.003443 | 0.00887 |
| comment_stylistics | Low | 12 | 0.01541 | 1.54% | 0.001285 | 0.00478 |
| cfg_complexity | Medium | 4 | 0.00690 | 0.69% | 0.001724 | 0.00552 |
| logic_flow | High | 1 | 0.00221 | 0.22% | 0.002209 | 0.00221 |
| syntactic_bias | Medium | 3 | 0.00095 | 0.10% | 0.000318 | 0.00056 |

### Bucket Share Changes

| Bucket | Previous | Current | Δ |
|---|---:|---:|---|
| structural_topology | 36.62% | 35.19% | −1.4 pp |
| ast_trigrams | 29.38% | 31.12% | +1.7 pp |
| semantic_fingerprint | 9.72% | 9.92% | +0.2 pp |
| comment_stylistics | 1.25% | 1.54% | +0.3 pp |
| cfg_complexity | 1.03% | 0.69% | −0.3 pp |

AST trigrams gained share — expected with more diverse AI code introducing new
structural patterns. Comment stylistics also gained, suggesting comment habits
differ between human and AI code.

### Privacy-Level Aggregation

| Privacy Level | Buckets | Features | Combined Share |
|---|---|---:|---:|
| **High** (safe for public attestation) | structural_topology, ast_trigrams, logic_flow | 274 | **66.5%** |
| **Medium** | semantic_fingerprint, layout_rhythm, cyclomatic_complexity, micro_stylistics, cfg_complexity, syntactic_bias | 213 | **21.6%** |
| **Low** (reveals coding style) | string_content, lexical_complexity, logical_idioms, comment_stylistics | 85 | **11.8%** |

**This finding remains robust**: Two-thirds of the model's discriminative power
comes from high-privacy features (structural topology + AST trigrams) that do not
reveal identifiable coding style. The privacy distribution is essentially
unchanged from the prior analysis, confirming Autograph's privacy-preserving
attestation design holds even with a more balanced dataset.

---

## 6. Cumulative Importance Concentration

| Threshold | Features Required | % of Total (572) |
|---:|---:|---:|
| 50% | 89 | 15.6% |
| 75% | 181 | 31.6% |
| 80% | 208 | 36.4% |
| 90% | 279 | 48.8% |
| 95% | 329 | 57.5% |
| 99% | 405 | 70.8% |

Half the total importance is concentrated in 89 features (15.6% of the feature
space), very close to the prior run (82 features, 14.5%). The long tail persists:
~170 features (30%) contribute < 5% combined.

### Near-Zero Importance Features

**36 features (6.3%)** have near-zero importance (< 1e-6), down from 40 (7.1%):

| Bucket | Near-Zero | Total | Pct |
|---|---:|---:|---:|
| string_content | 15 | 64 | 23.4% |
| semantic_fingerprint | 14 | 165 | 8.5% |
| cyclomatic_complexity | 6 | 31 | 19.4% |
| comment_stylistics | 1 | 12 | 8.3% |

The semantic_fingerprint bucket improved — fewer dead-weight features (14 vs 21
previously) as the expanded AI samples activated some previously-sparse
`call_freq_*` and `uses_*` columns.

---

## 7. Permutation Importance

Permutation importance measures the actual impact on test accuracy when a feature
is shuffled, avoiding MDI's known bias toward continuous features.

### Top 20 Features (Permutation)

| Rank | Feature | Importance | ±Std | Bucket |
|---:|---|---:|---:|---|
| 1 | `blank_line_ratio` | 0.02842 | 0.00537 | layout_rhythm |
| 2 | `max_consecutive_newlines` | 0.02000 | 0.00614 | layout_rhythm |
| 3 | `avg_vertical_chunk_size` | 0.01842 | 0.00539 | layout_rhythm |
| 4 | `quote_double` | 0.01737 | 0.00708 | micro_stylistics |
| 5 | `print_calls_ratio` | 0.01579 | 0.00000 | semantic_fingerprint |
| 6 | `node_keyword_argument` | 0.01474 | 0.00394 | structural_topology |
| 7 | `sentence_case_ratio` | 0.01474 | 0.00316 | comment_stylistics |
| 8 | `quote_single` | 0.01474 | 0.00698 | micro_stylistics |
| 9 | `node_import_statement` | 0.01158 | 0.00316 | structural_topology |
| 10 | `comment_to_code_ratio` | 0.01105 | 0.00598 | comment_stylistics |
| 11 | `call_freq_print` | 0.01053 | 0.00000 | semantic_fingerprint |
| 12 | `snake_case_ratio` | 0.01053 | 0.00333 | logical_idioms |
| 13 | `node_parameters` | 0.01000 | 0.00497 | structural_topology |
| 14 | `node_=` | 0.00947 | 0.00516 | structural_topology |
| 15 | `camel_case_words_ratio` | 0.00947 | 0.00567 | string_content |
| 16 | `method_call_ratio` | 0.00947 | 0.00211 | semantic_fingerprint |
| 17 | `node_import` | 0.00947 | 0.00211 | structural_topology |
| 18 | `node_function_definition` | 0.00895 | 0.00411 | structural_topology |
| 19 | `direct_import_ratio` | 0.00895 | 0.00474 | semantic_fingerprint |
| 20 | `node_class_definition` | 0.00895 | 0.00241 | structural_topology |

### Major Shift from Previous Run

The permutation importance landscape changed dramatically with the expanded
AI samples:

**Previously**: Dominated by micro_stylistics (`quote_single` #1) and specific
AST trigrams (subscript patterns, parse errors). These were niche signals that
separated small AI cohorts (3 test samples each) from humans.

**Now**: Layout rhythm claims the top 3 positions outright, and new semantic
signals emerged:

- **`print_calls_ratio`** (rank 5, std=0.000) and **`call_freq_print`** (rank 11)
  — print usage is now a zero-variance, maximally decisive signal. This strongly
  suggests AI-generated code uses `print()` at markedly different rates than
  human code — likely because AI models produce self-contained demo scripts that
  print outputs, while human production code uses logging frameworks.

- **`sentence_case_ratio`** (rank 7) and **`comment_to_code_ratio`** (rank 10)
  — comment stylistics leaped from weak (previously unranked in top 20) to
  top-10. AI models write comments differently: more "instructional" comments
  with sentence casing vs human developers' terse inline remarks.

- **`node_import_statement`** (rank 9), **`node_class_definition`** (rank 20),
  **`direct_import_ratio`** (rank 19) — import and class patterns are now decisive.
  AI-generated code favours specific import structures and may produce more
  class-heavy or more function-heavy code depending on the model.

- **`node_keyword_argument`** (rank 6) — keyword argument usage frequency
  separates coding styles, likely reflecting AI models' tendency toward explicit
  keyword arguments for "readable" generated code.

---

## 8. MDI vs Permutation — Agreement Analysis

### Top-20 Overlap

**7 of 20** features appear in both top-20 lists (up from 4/20):

| Feature | MDI Rank | Perm Rank | Bucket |
|---|---:|---:|---|
| `avg_vertical_chunk_size` | 1 | 3 | layout_rhythm |
| `max_consecutive_newlines` | 2 | 2 | layout_rhythm |
| `blank_line_ratio` | 3 | 1 | layout_rhythm |
| `quote_double` | 4 | 4 | micro_stylistics |
| `snake_case_ratio` | 5 | 12 | logical_idioms |
| `quote_single` | 7 | 8 | micro_stylistics |
| `camel_case_words_ratio` | 11 | 15 | string_content |

### Features Unique to Each Method

**MDI-only** (top 20): Dominated by lexical complexity and generic AST node
counts — continuous features with high cardinality that get many splits but
don't individually change predictions much:
- `identifier_entropy`, `avg_identifier_length`, `short_identifier_ratio`
- `avg_branching_factor`, `total_nodes`
- `node_identifier`, `node_.`, `node_attribute`, `node_,`
- `node_call`, `node_module`, `node_expression_statement`
- `camel_case_ratio`

**Permutation-only** (top 20): Dominated by semantic and structural signals
that decisively separate identity classes — especially AI vs human:
- `print_calls_ratio`, `call_freq_print` — print usage (AI marker)
- `sentence_case_ratio`, `comment_to_code_ratio` — comment patterns
- `node_keyword_argument`, `node_parameters` — argument style
- `node_import_statement`, `node_import`, `direct_import_ratio` — import patterns
- `node_=`, `node_function_definition`, `node_class_definition` — code structure
- `method_call_ratio` — API call patterns

### Interpretation

The improved overlap (7/20, up from 4/20) reflects a healthier dataset. With
the prior sparse AI samples, permutation importance was dominated by niche
signals that happened to perfectly separate the 3-sample AI test sets. With
50+ samples per AI identity, the permutation rankings now better reflect
genuine discriminative features.

1. **Both methods agree** on layout rhythm (top 3) and quote preference as the
   strongest signals. This convergence is meaningful — these features are both
   frequently used in splits (MDI) and genuinely predictive (permutation).

2. **Permutation reveals AI-vs-human markers** that MDI misses: print usage,
   comment style, and import patterns are critical for the now-learnable AI
   classes. MDI distributes their importance across thousands of tree splits,
   understating their true predictive contribution.

3. **MDI still over-values generic continuous features** like `identifier_entropy`
   and `total_nodes` that provide many splitting opportunities without decisive
   predictive power.

**For feature selection**, permutation importance should be preferred. The
improved agreement with MDI gives additional confidence in the top features.

---

## 9. Top 5 Features per Bucket

### layout_rhythm (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 1 | `avg_vertical_chunk_size` | 0.01535 |
| 2 | `max_consecutive_newlines` | 0.01421 |
| 3 | `blank_line_ratio` | 0.01341 |

All 3 features rank in both the MDI and permutation top 3. Layout rhythm now
**also dominates permutation importance** (previously permutation favoured
micro_stylistics). With AI samples in the mix, vertical spacing is the single
strongest discriminator: AI models produce code with distinctive whitespace
patterns that differ from human habits. This bucket has the **highest mean
importance** (0.0143) despite having only 3 features.

### micro_stylistics (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 4 | `quote_double` | 0.01012 |
| 7 | `quote_single` | 0.00856 |
| 30 | `quote_doubledoubledouble` | 0.00578 |
| 75 | `indent_width` | 0.00400 |
| 214 | `trailing_commas_count` | 0.00162 |

Quote preference remains a top signal. Notable shift: `quote_double` overtook
`quote_single` in both MDI (was rank 5, now 4) and permutation (now rank 4).
This likely reflects AI models' strong preference for double quotes in generated
Python code, making double-quote frequency a better discriminator with the
expanded AI dataset.

### lexical_complexity (Privacy: Low)

| Rank | Feature | MDI |
|---:|---|---:|
| 6 | `identifier_entropy` | 0.00861 |
| 10 | `avg_identifier_length` | 0.00764 |
| 13 | `short_identifier_ratio` | 0.00714 |

Naming diversity and verbosity remain strong author signals. AI models tend to
produce more verbose, descriptive variable names (`processed_result`,
`user_input`), while experienced human developers often use terser names in
context (`res`, `inp`).

### logical_idioms (Privacy: Low)

| Rank | Feature | MDI |
|---:|---|---:|
| 5 | `snake_case_ratio` | 0.00887 |
| 20 | `camel_case_ratio` | 0.00657 |
| 141 | `class_definition_count` | 0.00261 |
| 198 | `f_string_ratio` | 0.00181 |
| 373 | `try_except_count` | 0.00047 |

`camel_case_ratio` rose significantly (rank 36 → 20), suggesting AI models have
distinct naming convention patterns that differ from the human Python norm of
strict snake_case.

### structural_topology (Privacy: High)

| Rank | Feature | MDI |
|---:|---|---:|
| 8 | `avg_branching_factor` | 0.00825 |
| 9 | `node_identifier` | 0.00766 |
| 12 | `node_.` | 0.00737 |
| 14 | `node_attribute` | 0.00711 |
| 15 | `total_nodes` | 0.00701 |

`avg_branching_factor` rose from rank 15 to 8, becoming the top structural
feature. The branching structure of AST trees differs measurably between AI and
human code — AI models may produce more uniformly structured code with less
variation in branching depth.

### ast_trigrams (Privacy: High)

| Rank | Feature | MDI |
|---:|---|---:|
| 38 | `string:string_start:string_content` | 0.00530 |
| 40 | `expression_statement:string:string_start` | 0.00529 |
| 41 | `block:expression_statement:string` | 0.00527 |
| 42 | `string_start:string_content:string_end` | 0.00517 |
| 46 | `identifier:.:identifier` | 0.00508 |

String-related trigrams continue to dominate this bucket. The
`block:expression_statement:string` trigram (rank 41) captures docstring patterns
at block scope — a pattern where AI-generated code tends to produce more
docstrings than typical human code.

### semantic_fingerprint (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 36 | `call_freq_FastAPI` | 0.00538 |
| 37 | `uses_fastapi` | 0.00531 |
| 39 | `method_call_ratio` | 0.00530 |
| 56 | `keyword_arg_ratio` | 0.00467 |
| 61 | `call_freq_app_get` | 0.00441 |

FastAPI-specific features remain prominent as dataset artefacts (specific to 1–2
identities). However, `method_call_ratio` (rank 39 MDI, rank 16 permutation)
is a genuinely useful signal — the ratio of method calls to total calls captures
OOP-vs-procedural coding style.

### cyclomatic_complexity (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 21 | `decisions_per_line` | 0.00639 |
| 68 | `total_decision_points` | 0.00419 |
| 92 | `total_functions_analyzed` | 0.00349 |
| 100 | `avg_cyclomatic_complexity` | 0.00331 |
| 109 | `max_cyclomatic_complexity` | 0.00306 |

`decisions_per_line` remains the top complexity signal (rank 21). The overall
bucket share declined slightly (3.93% → 3.67%), suggesting complexity metrics
are somewhat less discriminative when AI models (which tend to produce
moderate-complexity code) are well-represented.

### comment_stylistics (Privacy: Low)

| Rank | Feature | MDI |
|---:|---|---:|
| 52 | `comment_to_code_ratio` | 0.00478 |
| 116 | `length_variance` | 0.00292 |
| 117 | `sentence_case_ratio` | 0.00290 |
| 224 | `dead_code_density` | 0.00157 |
| 262 | `inline_comment_ratio` | 0.00129 |

**Major improvement**: This bucket's share rose from 1.25% to 1.54%, and
`comment_to_code_ratio` jumped from rank 76 to 52 in MDI (and rank 10 in
permutation). `sentence_case_ratio` ranks #7 in permutation — AI models write
comments in sentence case, while human developers vary widely. This bucket is
now a key human-vs-AI discriminator.

### cfg_complexity (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 34 | `exit_density` | 0.00552 |
| 268 | `guard_clause_score` | 0.00124 |
| 458 | `while_true_ratio` | 0.00008 |
| 476 | `break_statement_count` | 0.00006 |

`exit_density` dropped significantly (rank 9 → 34). With AI code in the mix,
early-return patterns are less distinctive — AI models produce code with varying
exit patterns, diluting this signal's discriminative power.

### syntactic_bias (Privacy: Medium)

| Rank | Feature | MDI |
|---:|---|---:|
| 364 | `literal_collection_ratio` | 0.00056 |
| 383 | `exception_depth` | 0.00037 |
| 517 | `boolean_style_score` | 0.00002 |

Still the weakest bucket (0.10% of total). Unchanged — these features need
more diverse data or refinement.

### logic_flow (Privacy: High)

| Rank | Feature | MDI |
|---:|---|---:|
| 168 | `functional_score` | 0.00221 |

Dropped from rank 115 to 168. Functional vs procedural bias is less
discriminative when AI models (which produce both styles) are represented.

---

## 10. Actionable Recommendations

### Feature Selection

1. **Safe to prune**: The bottom ~170 features (30%) contribute < 5% total
   importance. Removing them would simplify the model with negligible accuracy loss.

2. **Candidate compact model**: 89 features capture 50% of importance. Combined
   with permutation-guided selection, a 60–100 feature model could match or
   exceed the full model's generalization (less overfitting).

3. **Dead features to remove**: 36 near-zero features (mostly sparse `call_freq_*`
   and `string_content` columns) are noise and should be dropped.

### Privacy Trade-offs

4. **High-privacy attestation remains viable**: Structural topology + AST
   trigrams provide 66.5% of discriminative signal, essentially unchanged from
   the prior analysis. This result is robust to dataset composition changes.

5. **Layout rhythm is a privacy bargain**: 3 features, Medium privacy, 4.3% of
   importance — and now **dominates permutation importance** (top 3). Adding
   these to the High-privacy set would boost signal substantially.

6. **Comment stylistics emerged as a human-vs-AI discriminator**: Consider
   adding `comment_to_code_ratio` and `sentence_case_ratio` to the attestation
   pipeline specifically for AI detection.

### Data Collection

7. ~~**AI identities need more samples**~~: **RESOLVED**. AI identities now have
   50–55 samples each. All five are learnable (F1 0.21–0.70). The sparse
   attestation problem that caused the prior three AI identities to score F1=0.00
   is fixed.

8. **DeepSeek V3 remains weakest** (F1 0.21): Its coding style may genuinely
   overlap with other AI models. Consider collecting more diverse DeepSeek
   prompts, or accepting that some AI models are fundamentally similar in
   style.

9. **Library-specific features (`uses_fastapi`, etc.) are still dataset
   artefacts**: With more diverse data, these will naturally diminish. Don't
   over-index on them.

### Model Improvements

10. **Try a reduced-feature RF or XGBoost**: Train on the top 80–100 features
    by permutation importance. This should reduce overfitting (current gap: 0.28
    between train and test accuracy, improved from 0.31).

11. **Cross-validate**: The current single 80/20 split gives 10–11 test samples
    for AI identities (up from 3). Results are more reliable but stratified
    k-fold (k=5) would still provide better estimates.

12. **Investigate print_calls_ratio**: Its zero-variance permutation importance
    (std=0.000) suggests it may be a near-perfect AI-vs-human separator.
    Validate on held-out data and consider featuring it in the binary
    human/AI classification pathway.

---

## Appendix: Reproducing This Analysis

```bash
cd autograph-ds

# Re-process dataset (if raw samples changed)
python process_dataset.py

# Re-train models
python train_models.py

# MDI only (fast, ~1s)
python report_feature_importance.py

# MDI + Permutation importance (~2 min)
python report_feature_importance.py --permutation

# Customize
python report_feature_importance.py --n-trees 2000 --top-n 50 --perm-repeats 20
```
