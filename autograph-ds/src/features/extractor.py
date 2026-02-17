import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import re

class LogicalDNAExtractor:
    def __init__(self):
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)

    def extract(self, code: str, enabled_buckets: List[str] = None) -> Dict[str, Any]:
        """
        Extracts features from Python code based on opted-in buckets.

        Args:
            code: Python source code to analyze
            enabled_buckets: List of feature buckets to extract (default: all)

        Returns:
            Dictionary of extracted features

        Raises:
            ValueError: If code is empty or parsing fails
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")

        if enabled_buckets is None:
            enabled_buckets = ["structural_topology", "micro_stylistics", "logical_idioms"]

        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            root_node = tree.root_node
        except Exception as e:
            raise ValueError(f"Failed to parse code: {e}")
        
        dna = {}
        
        if "structural_topology" in enabled_buckets:
            dna.update(self._extract_structural(root_node))
        
        if "micro_stylistics" in enabled_buckets:
            dna.update(self._extract_stylistic(code, root_node))
            
        if "logical_idioms" in enabled_buckets:
            dna.update(self._extract_idiomatic(code, root_node))
        
        return dna

    def _extract_structural(self, root_node: Node) -> Dict[str, Any]:
        node_types = []
        depths = []
        branching_factors = []

        def traverse(node: Node, depth: int):
            node_types.append(node.type)
            depths.append(depth)
            
            child_count = node.child_count
            if child_count > 0:
                branching_factors.append(child_count)
            
            for child in node.children:
                traverse(child, depth + 1)

        traverse(root_node, 0)
        
        counts = Counter(node_types)
        total_nodes = len(node_types)
        
        # Normalize counts
        node_type_dist = {k: v / total_nodes for k, v in counts.items()}
        
        return {
            "node_type_counts": node_type_dist,
            "max_nesting_depth": max(depths) if depths else 0,
            "avg_branching_factor": np.mean(branching_factors) if branching_factors else 0,
            "total_nodes": total_nodes
        }

    def _extract_stylistic(self, code: str, root_node: Node) -> Dict[str, Any]:
        lines = code.splitlines()
        
        # Indentation detection
        indent_widths = []
        indent_types = [] # 0 for space, 1 for tab
        
        for line in lines:
            if not line.strip():
                continue
            match = re.match(r'^(\s+)', line)
            if match:
                indent = match.group(1)
                if '\t' in indent:
                    indent_types.append(1)
                else:
                    indent_types.append(0)
                    indent_widths.append(len(indent))

        # Quote preference
        quotes = re.findall(r"(['\"]{1,3})", code)
        quote_counts = Counter(quotes)
        
        # Trailing commas in lists/dicts (simplified)
        trailing_commas = len(re.findall(r",\s*[\]\}]", code))
        
        return {
            "indent_type": np.mean(indent_types) if indent_types else 0.5, # ratio of tabs
            "indent_width": np.median(indent_widths) if indent_widths else 4,
            "quote_preference": dict(quote_counts),
            "trailing_commas_count": trailing_commas
        }

    def _extract_idiomatic(self, code: str, root_node: Node) -> Dict[str, Any]:
        # Naming convention (snake vs camel)
        identifiers = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", code)
        snake_case = [id for id in identifiers if "_" in id and id.islower()]
        camel_case = [id for id in identifiers if any(c.isupper() for c in id) and id[0].islower()]
        
        # f-strings vs .format
        f_strings = len(re.findall(r"f['\"]", code))
        format_calls = len(re.findall(r"\.format\(", code))
        
        # Comprehensions
        list_comps = len(re.findall(r"\[.*for.*in.*\]", code))
        
        return {
            "snake_case_ratio": len(snake_case) / len(identifiers) if identifiers else 0,
            "camel_case_ratio": len(camel_case) / len(identifiers) if identifiers else 0,
            "f_string_ratio": f_strings / (f_strings + format_calls) if (f_strings + format_calls) > 0 else 0.5,
            "list_comprehension_count": list_comps
        }
