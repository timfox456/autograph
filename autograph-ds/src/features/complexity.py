"""Cyclomatic complexity and advanced control flow features.

This module implements standard cyclomatic complexity metrics and additional
code quality measures that capture decision complexity.
"""

import re
from typing import Dict, Any
from collections import Counter
from tree_sitter import Node


class CyclomaticComplexityExtractor:
    """Extracts cyclomatic complexity and advanced control flow features."""

    # Decision nodes that increase cyclomatic complexity
    DECISION_NODE_TYPES = {
        "if_statement",
        "elif_clause",
        # 'else_clause' does not add a new path; exclude from CC
        "while_statement",
        "for_statement",
        "except_clause",
        # 'with_statement' is not a decision point; exclude
        "case_clause",  # match/case
        "conditional_expression",  # ternary
    }

    # Logical operators that add complexity
    LOGICAL_OPERATORS = {"and", "or"}

    def extract(self, code: str, root_node: Node) -> Dict[str, Any]:
        """Extract cyclomatic complexity and related metrics."""
        features = {}

        # Calculate standard cyclomatic complexity
        complexity_features = self._calculate_cyclomatic_complexity(code, root_node)
        features.update(complexity_features)

        # Advanced branching analysis
        branching_features = self._analyze_branching_patterns(code, root_node)
        features.update(branching_features)

        # Exception handling complexity
        exception_features = self._analyze_exception_complexity(code, root_node)
        features.update(exception_features)

        return features

    def _calculate_cyclomatic_complexity(self, code: str, root_node: Node) -> Dict[str, Any]:
        """
        Calculate McCabe cyclomatic complexity (10 features).

        CC = E - N + 2P, where:
        - E = number of edges
        - N = number of nodes
        - P = number of connected components

        Simplified: CC = 1 + number of decision points
        """
        features = {}

        # Count decision points per function
        function_complexities = []
        global_decisions = 0

        def analyze_function(node: Node, depth: int = 0) -> int:
            """Calculate complexity for a single function."""
            decisions = 0

            def traverse_decisions(n: Node):
                nonlocal decisions

                # Count decision nodes
                if n.type in self.DECISION_NODE_TYPES:
                    decisions += 1

                # Count logical operators in conditions
                if n.type in ("boolean_operator", "comparison_operator"):
                    decisions += 1

                # Count explicit boolean operations
                if n.type == "binary_operator":
                    text = self._get_node_text(n, code)
                    if text and ("and" in text or "or" in text):
                        decisions += 1

                for child in n.children:
                    traverse_decisions(child)

            traverse_decisions(node)
            return decisions

        # Find all function definitions
        def traverse_functions(node: Node):
            nonlocal global_decisions

            if node.type in ("function_definition", "method_definition"):
                func_decisions = analyze_function(node)
                function_complexities.append(func_decisions)
            else:
                # Count global-level decisions
                if node.type in self.DECISION_NODE_TYPES:
                    global_decisions += 1

            for child in node.children:
                traverse_functions(child)

        traverse_functions(root_node)

        # Calculate complexity metrics
        # Base complexity is 1, plus decisions
        total_complexities = [1 + d for d in function_complexities]
        if not total_complexities:
            total_complexities = [1 + global_decisions]

        features["avg_cyclomatic_complexity"] = sum(total_complexities) / len(total_complexities)
        features["max_cyclomatic_complexity"] = max(total_complexities) if total_complexities else 1
        features["min_cyclomatic_complexity"] = min(total_complexities) if total_complexities else 1
        features["total_functions_analyzed"] = len(function_complexities)

        # Complexity distribution
        low_complexity = sum(1 for c in total_complexities if c <= 10)
        medium_complexity = sum(1 for c in total_complexities if 11 <= c <= 20)
        high_complexity = sum(1 for c in total_complexities if c > 20)

        total_funcs = len(total_complexities) if total_complexities else 1
        features["low_complexity_ratio"] = low_complexity / total_funcs
        features["medium_complexity_ratio"] = medium_complexity / total_funcs
        features["high_complexity_ratio"] = high_complexity / total_funcs

        # Decision density
        total_lines = len(code.splitlines())
        total_decisions = sum(function_complexities) + global_decisions
        features["decisions_per_line"] = total_decisions / total_lines if total_lines > 0 else 0
        features["total_decision_points"] = total_decisions

        return features

    def _analyze_branching_patterns(self, code: str, root_node: Node) -> Dict[str, Any]:
        """
        Analyze branching patterns and styles (10 features).
        """
        features = {}

        # Count different branching structures
        if_counts = 0
        elif_counts = 0
        else_counts = 0
        nested_if_depths = []
        switch_like_patterns = 0

        def traverse_branches(node: Node, if_depth: int = 0):
            nonlocal if_counts, elif_counts, else_counts, switch_like_patterns

            if node.type == "if_statement":
                if_counts += 1
                new_depth = if_depth + 1
                nested_if_depths.append(new_depth)

                # Check for switch-like pattern (long if-elif chains)
                elif_count = len([c for c in node.children if c.type == "elif_clause"])
                if elif_count >= 3:
                    switch_like_patterns += 1

            elif node.type == "elif_clause":
                elif_counts += 1

            elif node.type == "else_clause":
                else_counts += 1

            for child in node.children:
                traverse_branches(child, if_depth if node.type != "if_statement" else if_depth + 1)

        traverse_branches(root_node)

        # Branching metrics
        total_branches = if_counts + elif_counts + else_counts
        features["if_count"] = if_counts
        features["elif_count"] = elif_counts
        features["else_count"] = else_counts
        features["average_if_depth"] = (
            sum(nested_if_depths) / len(nested_if_depths) if nested_if_depths else 0
        )
        features["max_if_nesting_depth"] = max(nested_if_depths) if nested_if_depths else 0
        features["elif_to_if_ratio"] = elif_counts / if_counts if if_counts > 0 else 0
        features["else_to_if_ratio"] = else_counts / if_counts if if_counts > 0 else 0
        features["switch_like_pattern_count"] = switch_like_patterns

        # Ternary operator usage
        ternary_count = len(re.findall(r"\bif\b.*\belse\b", code))
        features["ternary_operator_count"] = ternary_count
        features["ternary_to_branch_ratio"] = (
            ternary_count / total_branches if total_branches > 0 else 0
        )

        return features

    def _analyze_exception_complexity(self, code: str, root_node: Node) -> Dict[str, Any]:
        """
        Analyze exception handling complexity (10 features).
        """
        features = {}

        try_count = 0
        except_count = 0
        finally_count = 0
        bare_except_count = 0
        specific_exception_counts = Counter()

        def traverse_exceptions(node: Node):
            nonlocal try_count, except_count, finally_count, bare_except_count

            if node.type == "try_statement":
                try_count += 1

            elif node.type == "except_clause":
                except_count += 1

                # Check for bare except
                exception_type = node.child_by_field_name("type")
                if not exception_type:
                    bare_except_count += 1
                else:
                    # Count specific exception types
                    exc_text = self._get_node_text(exception_type, code)
                    if exc_text:
                        base_exc = exc_text.split(".")[-1].strip()
                        specific_exception_counts[base_exc] += 1

            elif node.type == "finally_clause":
                finally_count += 1

            for child in node.children:
                traverse_exceptions(child)

        traverse_exceptions(root_node)

        # Exception metrics
        features["try_block_count"] = try_count
        features["except_clause_count"] = except_count
        features["finally_clause_count"] = finally_count
        features["bare_except_ratio"] = bare_except_count / except_count if except_count > 0 else 0
        features["exception_handling_depth"] = except_count / try_count if try_count > 0 else 0

        # Most common exception types (top 5)
        top_exceptions = specific_exception_counts.most_common(5)
        for i, (exc_name, count) in enumerate(top_exceptions):
            features[f"exception_type_{exc_name}_count"] = count

        # Fill remaining exception slots
        for i in range(len(top_exceptions), 5):
            features[f"exception_type_placeholder_{i}"] = 0

        # Exception specificity score
        total_specific = sum(specific_exception_counts.values())
        features["specific_exception_ratio"] = (
            total_specific / except_count if except_count > 0 else 0
        )

        return features

    def _get_node_text(self, node: Node, code: str) -> str:
        """Extract text from a node."""
        try:
            start = node.start_byte
            end = node.end_byte
            return code[start:end]
        except Exception:
            return ""
