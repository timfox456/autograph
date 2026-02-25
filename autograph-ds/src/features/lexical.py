import numpy as np
import math
from collections import Counter
from typing import Dict, Any
from tree_sitter import Node


class LexicalComplexityExtractor:
    """
    Analyzes identifier naming traits, entropy, and 'mechanicalness'.
    """

    def extract(self, code: str, root_node: Node) -> Dict[str, Any]:
        identifiers = []

        def traverse(node: Node):
            if node.type == "identifier":
                name = node.text.decode("utf-8", errors="ignore")
                identifiers.append(name)
            for child in node.children:
                traverse(child)

        traverse(root_node)

        if not identifiers:
            return {
                "avg_identifier_length": 0,
                "short_identifier_ratio": 0,
                "identifier_entropy": 0,
            }

        lengths = [len(id) for id in identifiers]
        short_ids = [id for id in identifiers if len(id) <= 3]

        # Calculate character-level entropy across all identifiers
        all_chars = "".join(identifiers)
        entropy = self._calculate_entropy(all_chars)

        return {
            "avg_identifier_length": float(np.mean(lengths)),
            "short_identifier_ratio": len(short_ids) / len(identifiers),
            "identifier_entropy": entropy,
        }

    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0
        counts = Counter(text)
        probs = [c / len(text) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)
