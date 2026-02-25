"""String content analysis features for authorship attribution.

This module analyzes the semantic content of strings to capture:
- Error message patterns and style
- Docstring conventions
- String constant semantics (URLs, paths, SQL, etc.)
- Placeholder naming conventions
"""

import re
from typing import Dict, Any, List
from collections import Counter as CollectionsCounter
from tree_sitter import Node


class StringContentExtractor:
    """Extracts semantic patterns from string content."""

    # Error message verb patterns
    ERROR_VERBS = [
        "failed",
        "could not",
        "cannot",
        "can't",
        "unable to",
        "error",
        "invalid",
        "missing",
        "not found",
        "does not exist",
        "doesn't exist",
        "unsupported",
        "unexpected",
        "wrong",
        "bad",
        "corrupt",
        "broken",
        "timeout",
        "refused",
        "denied",
        "forbidden",
        "unauthorized",
        "not allowed",
        "abort",
        "fatal",
    ]

    # Docstring style markers
    DOCSTRING_STYLES = {
        "google": [r"Args:\s*\n", r"Returns:\s*\n", r"Raises:\s*\n", r"Yields:\s*\n"],
        "numpy": [r"Parameters\s*\n[-=]+", r"Returns\s*\n[-=]+", r"Raises\s*\n[-=]+"],
        "sphinx": [r":param\s+\w+:", r":return:", r":raise\s+\w+:", r":type\s+\w+:"],
        "epytext": [r"@param\s+\w+:", r"@return:", r"@raise\s+\w+:"],
    }

    # Semantic patterns in strings
    SEMANTIC_PATTERNS = {
        "url_pattern": r'https?://[^\s<>"{}|\\^`\[\]]+',
        "file_path": r'(?:\.\./|/|[a-zA-Z]:\\\\|\.\\)[^\s<>"|\\^`\[\]]+\.\w+',
        "email_pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "sql_keywords": r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN|GROUP|ORDER|LIMIT)\b",
        "regex_pattern": r"[\^\$\*\+\?\{\}\[\]\\\|\(\)\.\#\!]",
        "date_format": r"%[YmdHMS]",
        "hex_color": r"#[0-9A-Fa-f]{6}\b",
        "uuid_pattern": r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "version_string": r"\b\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?\b",
        "camel_case_words": r"\b[a-z]+[A-Z][a-zA-Z]*\b",
        "snake_case_words": r"\b[a-z]+_[a-z_]+\b",
    }

    # Placeholder naming patterns
    PLACEHOLDER_PATTERNS = {
        "anonymous_braces": r"\{\}",  # "{}"
        "numbered_braces": r"\{\d+\}",  # "{0}", "{1}"
        "named_braces": r"\{\w+\}",  # "{name}", "{value}"
        "percent_format": r"%[sdifouxXeEgGcrsa%]",  # "%s", "%d"
        "template_dollar": r"\$\w+",  # "$var"
        "template_curly": r"\$\{\w+\}",  # "${var}"
        "fstring_expr": r"\{[^}]+\}",  # f"{expr}" (will be caught by AST too)
    }

    def __init__(self):
        self.error_verbs = self.ERROR_VERBS
        self.docstring_styles = self.DOCSTRING_STYLES
        self.semantic_patterns = self.SEMANTIC_PATTERNS
        self.placeholder_patterns = self.PLACEHOLDER_PATTERNS

    def extract(self, code: str, root_node: Node) -> Dict[str, Any]:
        """Extract all string content features."""
        features = {}

        # Collect all string literals
        strings = self._extract_strings(code, root_node)
        total_strings = len(strings) if strings else 1

        # Error message analysis
        error_features = self._analyze_error_messages(strings, total_strings)
        features.update(error_features)

        # Docstring style analysis
        docstring_features = self._analyze_docstrings(code, strings, total_strings)
        features.update(docstring_features)

        # Semantic pattern analysis
        semantic_features = self._analyze_semantic_patterns(strings, total_strings)
        features.update(semantic_features)

        # Placeholder naming analysis
        placeholder_features = self._analyze_placeholders(strings, total_strings)
        features.update(placeholder_features)

        # String content metrics
        features["total_string_literals"] = len(strings)
        features["unique_string_count"] = len(set(strings))
        features["string_diversity"] = len(set(strings)) / len(strings) if strings else 0

        return features

    def _extract_strings(self, code: str, root_node: Node) -> List[str]:
        """Extract all string literal values from the code."""
        strings = []

        def traverse_strings(node: Node):
            if node.type in ("string", "string_content"):
                text = self._get_node_text(node, code)
                if text:
                    strings.append(text)

            for child in node.children:
                traverse_strings(child)

        traverse_strings(root_node)
        return strings

    def _analyze_error_messages(self, strings: List[str], total: int) -> Dict[str, Any]:
        """Analyze error message patterns (5 features)."""
        features = {}

        error_message_count = 0
        error_verb_counts = CollectionsCounter()

        for s in strings:
            s_lower = s.lower()
            is_error = False

            for verb in self.error_verbs:
                if verb in s_lower:
                    error_verb_counts[verb] += 1
                    is_error = True

            if is_error:
                error_message_count += 1

        # Error message ratio
        features["error_message_ratio"] = error_message_count / total

        # Most common error verbs (top 5)
        top_verbs = error_verb_counts.most_common(5)
        for verb, count in top_verbs:
            safe_verb = re.sub(r"[^\w]", "_", verb)
            features[f"error_verb_{safe_verb}_ratio"] = count / total

        # Fill remaining
        for i in range(len(top_verbs), 5):
            features[f"error_verb_placeholder_{i}"] = 0.0

        # Error verb diversity
        features["error_verb_diversity"] = len(error_verb_counts) / len(self.error_verbs)

        return features

    def _analyze_docstrings(self, code: str, strings: List[str], total: int) -> Dict[str, Any]:
        """Analyze docstring style conventions (5 features)."""
        features = {}

        docstring_counts = CollectionsCounter()

        for style_name, patterns in self.docstring_styles.items():
            count = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, code, re.IGNORECASE))
                count += matches

            if count > 0:
                docstring_counts[style_name] = count
                features[f"{style_name}_docstring_ratio"] = min(count / 10, 1.0)  # Normalize
            else:
                features[f"{style_name}_docstring_ratio"] = 0.0

        # Dominant style
        if docstring_counts:
            dominant_style = docstring_counts.most_common(1)[0][0]
            for style in self.docstring_styles.keys():
                features[f"is_{style}_style"] = 1 if style == dominant_style else 0
        else:
            # No docstrings detected
            for style in self.docstring_styles.keys():
                features[f"is_{style}_style"] = 0

        features["has_docstrings"] = 1 if docstring_counts else 0

        return features

    def _analyze_semantic_patterns(self, strings: List[str], total: int) -> Dict[str, Any]:
        """Analyze semantic patterns in string content (5 features)."""
        features = {}

        pattern_counts = CollectionsCounter()

        for s in strings:
            for pattern_name, pattern in self.semantic_patterns.items():
                matches = len(re.findall(pattern, s, re.IGNORECASE))
                if matches > 0:
                    pattern_counts[pattern_name] += matches

        # Normalize by total strings
        for pattern_name in self.semantic_patterns.keys():
            count = pattern_counts.get(pattern_name, 0)
            features[f"{pattern_name}_ratio"] = count / total

        # Semantic pattern diversity
        features["semantic_pattern_diversity"] = len(pattern_counts) / len(self.semantic_patterns)

        return features

    def _analyze_placeholders(self, strings: List[str], total: int) -> Dict[str, Any]:
        """Analyze placeholder naming conventions (5 features)."""
        features = {}

        placeholder_counts = CollectionsCounter()

        for s in strings:
            for placeholder_type, pattern in self.placeholder_patterns.items():
                matches = len(re.findall(pattern, s))
                if matches > 0:
                    placeholder_counts[placeholder_type] += matches

        # Normalize by total strings
        for placeholder_type in self.placeholder_patterns.keys():
            count = placeholder_counts.get(placeholder_type, 0)
            features[f"{placeholder_type}_ratio"] = count / total

        # Dominant placeholder style
        if placeholder_counts:
            dominant = placeholder_counts.most_common(1)[0][0]
            features["dominant_placeholder_style"] = list(self.placeholder_patterns.keys()).index(
                dominant
            )
        else:
            features["dominant_placeholder_style"] = -1

        # Placeholder naming diversity
        total_placeholders = sum(placeholder_counts.values())
        features["placeholder_diversity"] = (
            len(placeholder_counts) / len(self.placeholder_patterns)
            if self.placeholder_patterns
            else 0
        )
        features["placeholder_usage_ratio"] = total_placeholders / total

        return features

    def _get_node_text(self, node: Node, code: str) -> str:
        """Extract text from a node."""
        try:
            start = node.start_byte
            end = node.end_byte
            return code[start:end]
        except Exception:
            return ""
