from dataclasses import dataclass
from typing import List


@dataclass
class FeatureBucket:
    name: str
    description: str
    features: List[str]
    privacy_level: str  # 'High', 'Medium', 'Low'


# Bucket A: Structural Topology (High Privacy - Minimal PII)
STRUCTURAL_BUCKET = FeatureBucket(
    name="structural_topology",
    description="AST node distribution and control flow shape.",
    features=[
        "node_type_counts",
        "max_nesting_depth",
        "avg_branching_factor",
        "total_nodes",
    ],
    privacy_level="High",
)

# Bucket B: Micro-Stylistics (Medium Privacy - Formatting traits)
STYLISTIC_BUCKET = FeatureBucket(
    name="micro_stylistics",
    description="Indentation, quoting, and whitespace patterns.",
    features=[
        "indent_type",
        "indent_width",
        "quote_preference",
        "trailing_commas_count",
    ],
    privacy_level="Medium",
)

# Bucket C: Logical Idioms (Low Privacy - Author-specific patterns)
IDIOMATIC_BUCKET = FeatureBucket(
    name="logical_idioms",
    description="Naming conventions and preferred language constructs.",
    features=[
        "snake_case_ratio",
        "camel_case_ratio",
        "f_string_ratio",
        "list_comprehension_count",
        "try_except_count",
        "class_definition_count",
    ],
    privacy_level="Low",
)

# Bucket D: CFG Complexity (Medium Privacy - Execution flow patterns)
CFG_BUCKET = FeatureBucket(
    name="cfg_complexity",
    description="Early exit profiles, guard clauses, and loop interrupt motifs.",
    features=[
        "exit_density",
        "guard_clause_score",
        "while_true_ratio",
        "break_statement_count",
    ],
    privacy_level="Medium",
)

# Bucket E: Comment Stylistics (Low Privacy - Linguistic flavor)
COMMENT_BUCKET = FeatureBucket(
    name="comment_stylistics",
    description="Comment intent, variance, casing, and dead code signals.",
    features=[
        "comment_to_code_ratio",
        "instructional_ratio",
        "explanatory_ratio",
        "length_variance",
        "all_caps_ratio",
        "sentence_case_ratio",
        "dead_code_density",
        "emoji_density",
        "colorful_language_ratio",
    ],
    privacy_level="Low",
)

# Bucket F: AST Trigrams (High Privacy - Structural fingerprint)
TRIGRAM_BUCKET = FeatureBucket(
    name="ast_trigrams",
    description="Top-K structural node sequences.",
    features=[
        "top_trigrams",  # note: flattened into 500+ features during vectorization
    ],
    privacy_level="High",
)

# Bucket G: Layout Rhythm (Medium Privacy - Vertical spacing and layout)
LAYOUT_BUCKET = FeatureBucket(
    name="layout_rhythm",
    description="Vertical spacing, blank line density, and layout topology.",
    features=[
        "blank_line_ratio",
        "avg_vertical_chunk_size",
        "max_consecutive_newlines",
    ],
    privacy_level="Medium",
)

# Bucket H: Lexical Complexity (Low Privacy - Identifier naming traits)
LEXICAL_BUCKET = FeatureBucket(
    name="lexical_complexity",
    description="Identifier length, entropy, and short variable usage.",
    features=[
        "avg_identifier_length",
        "short_identifier_ratio",
        "identifier_entropy",
    ],
    privacy_level="Low",
)

# Bucket I: Syntactic Bias (Medium Privacy - Syntax and safety patterns)
SYNTACTIC_BUCKET = FeatureBucket(
    name="syntactic_bias",
    description="Literal vs constructor preference, boolean styles, and exception handling depth.",
    features=[
        "literal_collection_ratio",
        "boolean_style_score",
        "exception_depth",
    ],
    privacy_level="Medium",
)

# Bucket J: Logic Flow (High Privacy - Functional vs procedural bias)
FLOW_BUCKET = FeatureBucket(
    name="logic_flow",
    description="Functional primitives vs procedural loops.",
    features=[
        "functional_score",
    ],
    privacy_level="High",
)

ALL_BUCKETS = [
    STRUCTURAL_BUCKET,
    STYLISTIC_BUCKET,
    IDIOMATIC_BUCKET,
    CFG_BUCKET,
    COMMENT_BUCKET,
    TRIGRAM_BUCKET,
    LAYOUT_BUCKET,
    LEXICAL_BUCKET,
    SYNTACTIC_BUCKET,
    FLOW_BUCKET,
]
