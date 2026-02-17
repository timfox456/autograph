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
        "node_type_counts", "max_nesting_depth", 
        "avg_branching_factor", "cyclomatic_complexity"
    ],
    privacy_level="High"
)

# Bucket B: Micro-Stylistics (Medium Privacy - Formatting traits)
STYLISTIC_BUCKET = FeatureBucket(
    name="micro_stylistics",
    description="Indentation, quoting, and whitespace patterns.",
    features=[
        "indent_type", "indent_width", 
        "quote_preference", "trailing_commas"
    ],
    privacy_level="Medium"
)

# Bucket C: Logical Idioms (Low Privacy - Author-specific patterns)
IDIOMATIC_BUCKET = FeatureBucket(
    name="logical_idioms",
    description="Naming conventions and preferred language constructs.",
    features=[
        "naming_convention", "f_string_ratio", 
        "comprehension_vs_loop", "docstring_style"
    ],
    privacy_level="Low"
)

ALL_BUCKETS = [STRUCTURAL_BUCKET, STYLISTIC_BUCKET, IDIOMATIC_BUCKET]
