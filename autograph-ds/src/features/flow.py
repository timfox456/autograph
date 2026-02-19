import re
from typing import Dict, Any
from tree_sitter import Node

class LogicFlowExtractor:
    """
    Analyzes Functional vs Procedural bias.
    """

    def extract(self, code: str, root_node: Node) -> Dict[str, Any]:
        functional_markers = 0
        procedural_markers = 0

        # Functional: map, filter, lambda, comprehensions
        # Procedural: standard for/while loops, assignments in loops
        
        def traverse(node: Node):
            nonlocal functional_markers, procedural_markers
            
            if node.type in ('list_comprehension', 'set_comprehension', 'dictionary_comprehension', 'generator_expression', 'lambda'):
                functional_markers += 1
            
            if node.type == 'call':
                func = node.child_by_field_name('function')
                if func and func.text:
                    name = func.text.decode('utf-8', errors='ignore')
                    if name in ('map', 'filter', 'reduce', 'all', 'any', 'sorted', 'zip', 'enumerate', 'sum', 'min', 'max'):
                        functional_markers += 1

            if node.type in ('for_statement', 'while_statement'):
                procedural_markers += 1
            
            for child in node.children:
                traverse(child)

        traverse(root_node)

        total = functional_markers + procedural_markers
        return {
            "functional_score": functional_markers / total if total > 0 else 0.5
        }
