import numpy as np
from typing import Dict, Any
from tree_sitter import Node


class SyntacticBiasExtractor:
    """
    Analyzes literal vs constructor preference, boolean styles, and exception handling.
    """

    def extract(self, code: str, root_node: Node) -> Dict[str, Any]:
        stats = {
            "literals": 0,
            "constructors": 0,
            "bool_explicit": 0,  # if x is True / if x == True
            "bool_implicit": 0,  # if x
            "try_depths": [],
        }

        def traverse(node: Node, depth_in_try: int = 0):
            # Literals vs Constructors
            if node.type in ("list", "dictionary", "set"):
                stats["literals"] += 1
            elif node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.text.decode("utf-8", errors="ignore") in ("list", "dict", "set"):
                    stats["constructors"] += 1

            # Boolean Style
            if node.type == "if_statement":
                condition = node.child_by_field_name("condition")
                if condition:
                    cond_text = condition.text.decode("utf-8", errors="ignore")
                    if any(x in cond_text for x in ("is True", "is False", "== True", "== False")):
                        stats["bool_explicit"] += 1
                    else:
                        stats["bool_implicit"] += 1

            # Exception Depth
            is_try = node.type == "try_statement"
            if is_try:
                stats["try_depths"].append(depth_in_try + 1)

            for child in node.children:
                traverse(child, depth_in_try + (1 if is_try else 0))

        traverse(root_node)

        total_collections = stats["literals"] + stats["constructors"]
        total_bools = stats["bool_explicit"] + stats["bool_implicit"]

        return {
            "literal_collection_ratio": (
                stats["literals"] / total_collections if total_collections > 0 else 1.0
            ),
            "boolean_style_score": stats["bool_explicit"] / total_bools if total_bools > 0 else 0,
            "exception_depth": float(np.mean(stats["try_depths"])) if stats["try_depths"] else 0,
        }
