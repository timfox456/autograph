#!/usr/bin/env python3
"""Random Forest Feature Importance Analysis for Autograph Logical DNA.

Trains a large Random Forest on the processed dataset and produces a detailed
feature importance report, grouped by feature bucket/category.

Usage:
    python report_feature_importance.py
    python report_feature_importance.py --n-trees 2000 --top-n 50
    python report_feature_importance.py --permutation  # include permutation importance (slower)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).parent / "research/data/processed/dataset.csv"
META_COLUMNS = ["label", "identity", "filename"]


# ---------------------------------------------------------------------------
# Feature-to-bucket classification
# ---------------------------------------------------------------------------

def classify_feature(name: str) -> str:
    """Map a flat feature column name to its Logical DNA bucket."""
    # AST node type distribution (structural_topology)
    if name.startswith("node_"):
        return "structural_topology"

    # AST trigrams
    if name.startswith("top_trigrams_"):
        return "ast_trigrams"

    # Semantic fingerprint: API call frequencies + library usage flags
    if name.startswith("call_freq_") or name.startswith("uses_"):
        return "semantic_fingerprint"

    # Semantic fingerprint: import pattern ratios + domain ratios
    if name.endswith("_domain_ratio") or name in (
        "direct_import_ratio", "from_import_ratio", "star_import_ratio",
        "stdlib_ratio", "third_party_ratio", "unique_library_count",
        "method_call_ratio", "chained_call_ratio", "print_calls_ratio",
        "len_calls_ratio", "range_calls_ratio", "enumerate_calls_ratio",
        "zip_calls_ratio", "isinstance_calls_ratio", "hasattr_calls_ratio",
        "getattr_calls_ratio", "open_calls_ratio", "join_calls_ratio",
        "keyword_arg_ratio", "call_vocabulary_size", "call_diversity",
    ):
        return "semantic_fingerprint"

    # Cyclomatic complexity bucket
    if name in (
        "avg_cyclomatic_complexity", "max_cyclomatic_complexity",
        "min_cyclomatic_complexity", "total_functions_analyzed",
        "low_complexity_ratio", "medium_complexity_ratio",
        "high_complexity_ratio", "decisions_per_line",
        "total_decision_points",
    ):
        return "cyclomatic_complexity"

    # Decision-flow features (cyclomatic / cfg overlap)
    if name in (
        "if_count", "elif_count", "else_count",
        "average_if_depth", "max_if_nesting_depth",
        "elif_to_if_ratio", "else_to_if_ratio",
        "switch_like_pattern_count",
        "ternary_operator_count", "ternary_to_branch_ratio",
    ):
        return "cyclomatic_complexity"

    # Exception handling features
    if name.startswith("exception_type_") or name in (
        "try_block_count", "except_clause_count", "finally_clause_count",
        "bare_except_ratio", "exception_handling_depth",
        "specific_exception_ratio", "error_message_ratio",
    ):
        return "cyclomatic_complexity"

    # Error verb features
    if name.startswith("error_verb_"):
        return "string_content"

    # String content bucket
    if name.startswith("google_docstring_") or name.startswith("numpy_docstring_"):
        return "string_content"
    if name.startswith("sphinx_docstring_") or name.startswith("epytext_docstring_"):
        return "string_content"
    if name.startswith("is_google_") or name.startswith("is_numpy_"):
        return "string_content"
    if name.startswith("is_sphinx_") or name.startswith("is_epytext_"):
        return "string_content"
    if name in (
        "url_pattern_ratio", "file_path_ratio", "email_pattern_ratio",
        "sql_keywords_ratio", "regex_pattern_ratio", "date_format_ratio",
        "hex_color_ratio", "uuid_pattern_ratio", "version_string_ratio",
        "camel_case_words_ratio", "snake_case_words_ratio",
        "semantic_pattern_diversity",
        "anonymous_braces_ratio", "numbered_braces_ratio",
        "named_braces_ratio", "percent_format_ratio",
        "template_dollar_ratio", "template_curly_ratio",
        "fstring_expr_ratio", "dominant_placeholder_style",
        "placeholder_diversity", "placeholder_usage_ratio",
        "total_string_literals", "unique_string_count", "string_diversity",
    ):
        return "string_content"

    # Micro-stylistics
    if name in (
        "indent_type", "indent_width", "trailing_commas_count",
    ) or name.startswith("quote_"):
        return "micro_stylistics"

    # Logical idioms
    if name in (
        "snake_case_ratio", "camel_case_ratio", "f_string_ratio",
        "list_comprehension_count", "try_except_count",
        "class_definition_count",
    ):
        return "logical_idioms"

    # CFG complexity
    if name in (
        "exit_density", "guard_clause_score",
        "while_true_ratio", "break_statement_count",
    ):
        return "cfg_complexity"

    # Comment stylistics
    if name in (
        "comment_to_code_ratio", "instructional_ratio",
        "explanatory_ratio", "length_variance",
        "all_caps_ratio", "sentence_case_ratio",
        "dead_code_density", "emoji_density",
        "colorful_language_ratio", "inline_comment_ratio",
        "has_docstrings", "debt_markers_count",
    ):
        return "comment_stylistics"

    # Layout rhythm
    if name in (
        "blank_line_ratio", "avg_vertical_chunk_size",
        "max_consecutive_newlines",
    ):
        return "layout_rhythm"

    # Lexical complexity
    if name in (
        "avg_identifier_length", "short_identifier_ratio",
        "identifier_entropy",
    ):
        return "lexical_complexity"

    # Syntactic bias
    if name in (
        "literal_collection_ratio", "boolean_style_score",
        "exception_depth",
    ):
        return "syntactic_bias"

    # Logic flow
    if name == "functional_score":
        return "logic_flow"

    # Structural topology (remaining scalar features)
    if name in (
        "max_nesting_depth", "avg_branching_factor", "total_nodes",
    ):
        return "structural_topology"

    return "other"


BUCKET_PRIVACY = {
    "structural_topology": "High",
    "ast_trigrams": "High",
    "logic_flow": "High",
    "micro_stylistics": "Medium",
    "cfg_complexity": "Medium",
    "layout_rhythm": "Medium",
    "syntactic_bias": "Medium",
    "logical_idioms": "Low",
    "comment_stylistics": "Low",
    "lexical_complexity": "Low",
    "semantic_fingerprint": "Medium",
    "cyclomatic_complexity": "Medium",
    "string_content": "Low",
    "other": "Unknown",
}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        print("Dataset not found. Please run process_dataset.py first.")
        sys.exit(1)
    df = pd.read_csv(DATASET_PATH)
    # Filter identities with < 2 samples (can't stratify)
    counts = df["identity"].value_counts()
    valid_ids = counts[counts >= 2].index
    df = df[df["identity"].isin(valid_ids)]
    return df


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    random_state: int = 42,
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
        oob_score=True,
    )
    rf.fit(X_train, y_train)
    return rf


def build_importance_df(
    feature_names: list[str],
    importances: np.ndarray,
    std: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a sorted DataFrame of feature importances with bucket labels."""
    data = {
        "feature": feature_names,
        "importance": importances,
        "bucket": [classify_feature(f) for f in feature_names],
        "privacy": [BUCKET_PRIVACY.get(classify_feature(f), "Unknown") for f in feature_names],
    }
    if std is not None:
        data["std"] = std
    df = pd.DataFrame(data).sort_values("importance", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    df.index.name = "rank"
    return df


def print_section(title: str, char: str = "=") -> None:
    print()
    print(char * 70)
    print(title)
    print(char * 70)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_dataset_summary(df: pd.DataFrame, feature_names: list[str]) -> None:
    print_section("DATASET SUMMARY")
    print(f"  Samples:       {len(df)}")
    print(f"  Features:      {len(feature_names)}")
    print(f"  Identities:    {df['identity'].nunique()}")
    print(f"  Label split:   {df['label'].value_counts().to_dict()}")
    print()
    print("  Identity distribution:")
    for identity, count in df["identity"].value_counts().items():
        print(f"    {identity:20s} {count:4d} samples")


def report_model_summary(
    rf: RandomForestClassifier,
    n_estimators: int,
    train_time: float,
    train_acc: float,
    test_acc: float,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> None:
    print_section("RANDOM FOREST MODEL")
    print(f"  Trees:              {n_estimators}")
    print(f"  OOB Score:          {rf.oob_score_:.4f}")
    print(f"  Train Accuracy:     {train_acc:.4f}")
    print(f"  Test Accuracy:      {test_acc:.4f}")
    print(f"  Training Time:      {train_time:.1f}s")
    print()
    print("  Classification Report (test set):")
    print(classification_report(y_test, y_pred, zero_division=0))


def report_top_features(imp_df: pd.DataFrame, top_n: int, label: str) -> None:
    print_section(f"TOP {top_n} FEATURES — {label}")
    cumulative = 0.0
    total = imp_df["importance"].sum()
    for rank, row in imp_df.head(top_n).iterrows():
        cumulative += row["importance"]
        pct = row["importance"] / total * 100 if total > 0 else 0
        cum_pct = cumulative / total * 100 if total > 0 else 0
        std_str = f"  (±{row['std']:.5f})" if "std" in row.index else ""
        print(
            f"  {rank:3d}. {row['feature']:55s} "
            f"{row['importance']:.5f}{std_str}  "
            f"[{pct:5.2f}%  cum {cum_pct:5.1f}%]  "
            f"({row['bucket']})"
        )


def report_bucket_importance(imp_df: pd.DataFrame) -> None:
    print_section("IMPORTANCE BY FEATURE BUCKET")
    bucket_stats = (
        imp_df.groupby("bucket")
        .agg(
            total_importance=("importance", "sum"),
            feature_count=("importance", "count"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
        )
        .sort_values("total_importance", ascending=False)
    )
    total = imp_df["importance"].sum()
    print()
    print(f"  {'Bucket':<25s} {'Privacy':<8s} {'#Feat':>6s} {'Total':>10s} {'Pct':>7s} {'Mean':>10s} {'Max':>10s}")
    print(f"  {'-'*24:<25s} {'-'*7:<8s} {'-'*6:>6s} {'-'*10:>10s} {'-'*7:>7s} {'-'*10:>10s} {'-'*10:>10s}")
    for bucket, row in bucket_stats.iterrows():
        pct = row["total_importance"] / total * 100 if total > 0 else 0
        privacy = BUCKET_PRIVACY.get(bucket, "?")
        print(
            f"  {bucket:<25s} {privacy:<8s} {row['feature_count']:6.0f} "
            f"{row['total_importance']:10.5f} {pct:6.2f}% "
            f"{row['mean_importance']:10.6f} {row['max_importance']:10.5f}"
        )


def report_cumulative_thresholds(imp_df: pd.DataFrame) -> None:
    print_section("CUMULATIVE IMPORTANCE THRESHOLDS")
    total = imp_df["importance"].sum()
    cumulative = 0.0
    thresholds = [50, 75, 80, 90, 95, 99]
    reached = {}
    for rank, row in imp_df.iterrows():
        cumulative += row["importance"]
        pct = cumulative / total * 100 if total > 0 else 0
        for t in thresholds:
            if t not in reached and pct >= t:
                reached[t] = rank
    print()
    for t in thresholds:
        n = reached.get(t, len(imp_df))
        print(f"  {t:3d}% of total importance captured by top {n:4d} features (of {len(imp_df)})")


def report_top_per_bucket(imp_df: pd.DataFrame, per_bucket_n: int = 5) -> None:
    print_section(f"TOP {per_bucket_n} FEATURES PER BUCKET")
    for bucket in imp_df["bucket"].unique():
        bucket_df = imp_df[imp_df["bucket"] == bucket].head(per_bucket_n)
        if bucket_df.empty:
            continue
        privacy = BUCKET_PRIVACY.get(bucket, "?")
        print(f"\n  [{bucket}] (privacy: {privacy})")
        for rank, row in bucket_df.iterrows():
            print(f"    {rank:4d}. {row['feature']:50s} {row['importance']:.5f}")


def report_near_zero_features(imp_df: pd.DataFrame) -> None:
    print_section("NEAR-ZERO IMPORTANCE FEATURES")
    zero_mask = imp_df["importance"] < 1e-6
    n_zero = zero_mask.sum()
    total = len(imp_df)
    print(f"\n  {n_zero} of {total} features ({n_zero/total*100:.1f}%) have near-zero importance.")
    if n_zero > 0:
        by_bucket = imp_df[zero_mask].groupby("bucket").size().sort_values(ascending=False)
        print("\n  Near-zero features by bucket:")
        for bucket, count in by_bucket.items():
            bucket_total = (imp_df["bucket"] == bucket).sum()
            print(f"    {bucket:<25s} {count:4d} / {bucket_total:4d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Random Forest feature importance analysis for Autograph Logical DNA"
    )
    parser.add_argument(
        "--n-trees", type=int, default=1000,
        help="Number of trees in the Random Forest (default: 1000)"
    )
    parser.add_argument(
        "--top-n", type=int, default=40,
        help="Number of top features to display (default: 40)"
    )
    parser.add_argument(
        "--permutation", action="store_true",
        help="Also compute permutation importance (slower but more robust)"
    )
    parser.add_argument(
        "--perm-repeats", type=int, default=10,
        help="Number of repeats for permutation importance (default: 10)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test set fraction (default: 0.2)"
    )
    args = parser.parse_args()

    # --- Load data ---
    df = load_dataset()
    X = df.drop(columns=META_COLUMNS)
    y = df["identity"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    report_dataset_summary(df, feature_names)

    # --- Train large RF ---
    print(f"\nTraining Random Forest with {args.n_trees} trees ...")
    t0 = time.time()
    rf = train_rf(X_train, y_train, n_estimators=args.n_trees)
    train_time = time.time() - t0

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    y_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    report_model_summary(rf, args.n_trees, train_time, train_acc, test_acc, y_test, y_pred)

    # --- MDI (Mean Decrease Impurity) importance ---
    mdi_df = build_importance_df(feature_names, rf.feature_importances_)
    report_top_features(mdi_df, args.top_n, "Mean Decrease Impurity (MDI)")
    report_bucket_importance(mdi_df)
    report_cumulative_thresholds(mdi_df)
    report_top_per_bucket(mdi_df)
    report_near_zero_features(mdi_df)

    # --- Permutation importance (optional) ---
    if args.permutation:
        print(f"\nComputing permutation importance ({args.perm_repeats} repeats) ...")
        t0 = time.time()
        perm_result = permutation_importance(
            rf, X_test, y_test,
            n_repeats=args.perm_repeats,
            random_state=42,
            n_jobs=-1,
        )
        perm_time = time.time() - t0
        print(f"  Permutation importance computed in {perm_time:.1f}s")

        perm_df = build_importance_df(
            feature_names,
            perm_result.importances_mean,
            perm_result.importances_std,
        )
        report_top_features(perm_df, args.top_n, "Permutation Importance")

        # Compare MDI vs Permutation top features
        print_section("MDI vs PERMUTATION — TOP 20 COMPARISON")
        mdi_top = set(mdi_df.head(20)["feature"])
        perm_top = set(perm_df.head(20)["feature"])
        overlap = mdi_top & perm_top
        print(f"\n  Overlap in top 20: {len(overlap)} / 20")
        print(f"  MDI-only:          {mdi_top - perm_top}")
        print(f"  Permutation-only:  {perm_top - mdi_top}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
