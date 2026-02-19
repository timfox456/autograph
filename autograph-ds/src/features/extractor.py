import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import re
from .comments import CommentExtractor
from .layout import LayoutExtractor
from .lexical import LexicalComplexityExtractor
from .syntactic import SyntacticBiasExtractor
from .flow import LogicFlowExtractor

class LogicalDNAExtractor:
    def __init__(self):
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)
        self.comment_extractor = CommentExtractor()
        self.layout_extractor = LayoutExtractor()
        self.lexical_extractor = LexicalComplexityExtractor()
        self.syntactic_extractor = SyntacticBiasExtractor()
        self.flow_extractor = LogicFlowExtractor()

    def extract(self, code: str, enabled_buckets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extracts features from Python code based on opted-in buckets.

        Args:
            code: Python source code to analyze
            enabled_buckets: List of feature buckets to extract (default: all)

        Returns:
            Dictionary of extracted features

        Raises:
            ValueError: If code is empty or fails to parse
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")

        if enabled_buckets is None:
            enabled_buckets = [
                "structural_topology", 
                "micro_stylistics", 
                "logical_idioms",
                "cfg_complexity",
                "comment_stylistics",
                "ast_trigrams",
                "layout_rhythm",
                "lexical_complexity",
                "syntactic_bias",
                "logic_flow"
            ]

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

        if "cfg_complexity" in enabled_buckets:
            dna.update(self._extract_cfg_complexity(code, root_node))

        if "comment_stylistics" in enabled_buckets:
            dna.update(self.comment_extractor.extract(code, root_node))

        if "ast_trigrams" in enabled_buckets:
            dna["top_trigrams"] = self._extract_trigrams(root_node)
            
        if "layout_rhythm" in enabled_buckets:
            dna.update(self.layout_extractor.extract(code))
            
        if "lexical_complexity" in enabled_buckets:
            dna.update(self.lexical_extractor.extract(code, root_node))
            
        if "syntactic_bias" in enabled_buckets:
            dna.update(self.syntactic_extractor.extract(code, root_node))
            
        if "logic_flow" in enabled_buckets:
            dna.update(self.flow_extractor.extract(code, root_node))
        
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
        # Ensure ' and " are always present to avoid KeyErrors in tests/models
        quote_prefs = {
            "'": quote_counts.get("'", 0),
            '"': quote_counts.get('"', 0),
            "'''": quote_counts.get("'''", 0),
            '"""': quote_counts.get('"""', 0)
        }
        
        # Trailing commas in lists/dicts (simplified)
        trailing_commas = len(re.findall(r",\s*[\]\}]", code))
        
        return {
            "indent_type": np.mean(indent_types) if indent_types else 0.5, # ratio of tabs
            "indent_width": np.median(indent_widths) if indent_widths else 4,
            "quote_preference": quote_prefs,
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
        
        # More structural features for idiomatic bucket
        try_excepts = len(re.findall(r"\btry\b", code))
        classes = len(re.findall(r"\bclass\b", code))
        
        return {
            "snake_case_ratio": len(snake_case) / len(identifiers) if identifiers else 0,
            "camel_case_ratio": len(camel_case) / len(identifiers) if identifiers else 0,
            "f_string_ratio": f_strings / (f_strings + format_calls) if (f_strings + format_calls) > 0 else 0.5,
            "list_comprehension_count": list_comps,
            "try_except_count": try_excepts,
            "class_definition_count": classes
        }

    def _extract_cfg_complexity(self, code: str, root_node: Node) -> Dict[str, Any]:
        """
        Analyzes Control Flow Graph complexity markers.
        """
        early_exits = []
        guard_clauses = 0
        while_trues = 0
        while_total = 0
        breaks = []

        def traverse(node: Node, in_func: bool = False, func_start_line: int = -1):
            nonlocal while_trues, while_total, guard_clauses
            
            # Identify early exits inside functions
            if node.type in ('return_statement', 'raise_statement', 'continue_statement'):
                if in_func:
                    early_exits.append(node)
                    # Guard clause heuristic: exit in the first 5 lines of a function
                    if 0 <= node.start_point[0] - func_start_line < 5:
                        guard_clauses += 1
            
            if node.type == 'break_statement':
                breaks.append(node)

            if node.type == 'while_statement':
                while_total += 1
                # Check for 'while True:'
                condition = node.child_by_field_name('condition')
                if condition and condition.type == 'true':
                    while_trues += 1

            # Recurse with context
            is_func = node.type == 'function_definition'
            new_func_start = node.start_point[0] if is_func else func_start_line
            
            for child in node.children:
                traverse(child, in_func=(in_func or is_func), func_start_line=new_func_start)

        traverse(root_node)
        
        total_loc = len(code.splitlines())
        
        # Guard clause heuristic: exits that happen early in the function body
        exit_density = len(early_exits) / total_loc if total_loc > 0 else 0
        
        return {
            "exit_density": exit_density,
            "guard_clause_score": guard_clauses,
            "while_true_ratio": while_trues / while_total if while_total > 0 else 0,
            "break_statement_count": len(breaks)
        }

    def _extract_trigrams(self, root_node: Node) -> Dict[str, int]:
        """
        Extracts structural AST trigrams (sequences of 3 node types).
        """
        node_types = []

        def traverse(node: Node):
            # Filter out generic/noise nodes if needed, but for now take all
            node_types.append(node.type)
            for child in node.children:
                traverse(child)

        traverse(root_node)
        
        trigrams = []
        for i in range(len(node_types) - 2):
            trigrams.append(tuple(node_types[i:i+3]))
        
        # Return a counter of stringified trigrams
        counts = Counter([f"{t[0]}:{t[1]}:{t[2]}" for t in trigrams])
        return dict(counts)
