import re
import numpy as np
from collections import Counter
from tree_sitter import Node
from typing import Dict, Any, List

class CommentExtractor:
    """
    Analyzes the 'Social Layer' of the code: comments and docstrings.
    Distinguishes between human 'monologue' and AI 'instruction'.
    """
    
    # Python keywords to detect commented-out code (Ghost Code)
    PY_KEYWORDS = {'def', 'import', 'from', 'if', 'else', 'elif', 'while', 'for', 'return', 'class', 'try', 'except', 'with', 'yield'}

    # Common colorful/informal language (Human tell)
    COLORFUL_WORDS = {
        'wtf', 'hack', 'ugly', 'stupid', 'fixme', 'todo', 'shit', 'damn', 'hell', 
        'magic', 'voodoo', 'weird', 'temporary', 'shrug', 'guilty', 'sorry'
    }

    def __init__(self):
        # Instructional: Imperative verbs or "This function/script ..." (common in AI)
        self.instruction_patterns = [
            r'^(Calculate|Process|Return|Initialize|Get|Set|Create|Update|Delete|Check|Find)\b',
            r'^This (function|script|code|module) (calculates|provides|implements|handles)\b'
        ]
        # Explanatory: Causal conjunctions (common in humans)
        self.causal_conjunctions = r'\b(because|since|so that|due to|reason|as a result)\b'
        # Emoji regex: matches most common emojis
        self.emoji_regex = r'[\U00010000-\U0010ffff\u2600-\u27bf]'

    def extract(self, code: str, root_node: Node) -> Dict[str, Any]:
        lines = code.splitlines()
        total_loc = len(lines)
        
        comments = []
        docstrings = []
        
        # 1. Collect all comment nodes and docstrings
        def traverse(node: Node):
            if node.type == 'comment':
                comments.append(node)
            # Docstrings are often the first child of a block inside function_definition/class_definition
            elif node.type == 'string' and node.parent and node.parent.type == 'expression_statement':
                if node.parent.parent and node.parent.parent.type == 'block':
                    # Check if it's the first child of the block
                    if node.parent.parent.children[0] == node.parent:
                        docstrings.append(node)
            
            for child in node.children:
                traverse(child)

        traverse(root_node)
        
        comment_texts = [c.text.decode('utf-8', errors='ignore').strip('#').strip() for c in comments]
        comment_lengths = [len(t) for t in comment_texts if t]

        # 2. Stylometry
        # We check debt markers on the texts BEFORE stripping prefixes if those prefixes ARE markers
        debt_markers_count = sum(1 for t in comment_texts if any(m in t.upper() for m in ['TODO', 'FIXME', 'HACK']))
        
        # Emoji detection
        total_emojis = sum(len(re.findall(self.emoji_regex, t)) for t in comment_texts)
        
        # Colorful language detection
        colorful_count = 0
        for t in comment_texts:
            words = set(re.findall(r'\b\w+\b', t.lower()))
            if words.intersection(self.COLORFUL_WORDS):
                colorful_count += 1
        
        # Strip common prefixes like 'INSTRUCTIONAL:', 'COMMENT:', etc. for NLP-style analysis
        def clean_comment(t):
            # Don't strip if it looks like a debt marker
            if any(m in t.upper() for m in ['TODO', 'FIXME', 'HACK']):
                return t
            t = re.sub(r'^[A-Z]{3,}:', '', t).strip()
            return t

        cleaned_texts = [clean_comment(t) for t in comment_texts]
        
        casing = self._analyze_casing(cleaned_texts)
        intent = self._analyze_intent(cleaned_texts)
        dead_code_count = self._detect_dead_code(cleaned_texts)
        
        # 3. Placement (Inline vs Block)
        inline_count = 0
        for comment in comments:
            start_row = comment.start_point[0]
            if start_row < total_loc:
                line_content = lines[start_row].strip()
                # If the comment is not at the start of the line, it's inline
                if not line_content.startswith('#'):
                    inline_count += 1
        
        return {
            "comment_to_code_ratio": len(comments) / total_loc if total_loc > 0 else 0,
            "instructional_ratio": intent['instructional'] / len(comments) if comments else 0,
            "explanatory_ratio": intent['explanatory'] / len(comments) if comments else 0,
            "length_variance": float(np.var(comment_lengths)) if comment_lengths else 0,
            "all_caps_ratio": casing['all_caps'] / len(comments) if comments else 0,
            "sentence_case_ratio": casing['sentence_case'] / len(comments) if comments else 0,
            "dead_code_density": dead_code_count / len(comments) if comments else 0,
            "emoji_density": total_emojis / len(comments) if comments else 0,
            "colorful_language_ratio": colorful_count / len(comments) if comments else 0,
            "inline_comment_ratio": inline_count / len(comments) if comments else 0,
            "has_docstrings": len(docstrings) > 0,
            "debt_markers_count": debt_markers_count
        }

    def _analyze_casing(self, texts: List[str]) -> Dict[str, int]:
        stats = {'all_caps': 0, 'sentence_case': 0, 'lowercase': 0}
        for t in texts:
            if not t: continue
            if t.isupper() and len(t) > 3:
                stats['all_caps'] += 1
            elif t[0].isupper() and any(c.islower() for c in t):
                # Simple check for Sentence case (Starts with upper, contains lower)
                stats['sentence_case'] += 1
            elif t.islower():
                stats['lowercase'] += 1
        return stats

    def _analyze_intent(self, texts: List[str]) -> Dict[str, int]:
        stats = {'instructional': 0, 'explanatory': 0}
        for t in texts:
            if any(re.search(p, t, re.IGNORECASE) for p in self.instruction_patterns):
                stats['instructional'] += 1
            if re.search(self.causal_conjunctions, t, re.IGNORECASE):
                stats['explanatory'] += 1
        return stats

    def _detect_dead_code(self, texts: List[str]) -> int:
        count = 0
        for t in texts:
            # Look for multiple keywords or assignment
            words = set(re.findall(r'\b\w+\b', t))
            if words.intersection(self.PY_KEYWORDS) or '=' in t:
                # Heuristic: if it has '=' or more than one keyword
                if len(words.intersection(self.PY_KEYWORDS)) >= 1 or '=' in t:
                    count += 1
        return count
