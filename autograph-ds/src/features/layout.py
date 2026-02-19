import numpy as np
from typing import Dict, Any

class LayoutExtractor:
    """
    Analyzes the 'Vertical Rhythm' and layout topology of the code.
    Humans and AI models have distinct habits in how they use whitespace.
    """

    def extract(self, code: str) -> Dict[str, Any]:
        lines = code.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return {
                "blank_line_ratio": 0,
                "avg_vertical_chunk_size": 0,
                "max_consecutive_newlines": 0
            }

        blank_lines = [i for i, line in enumerate(lines) if not line.strip()]
        num_blank = len(blank_lines)
        
        # Calculate vertical chunks (groups of code lines between blank lines)
        chunks = []
        current_chunk = 0
        max_consecutive = 0
        consecutive = 0
        
        for line in lines:
            if not line.strip():
                if current_chunk > 0:
                    chunks.append(current_chunk)
                    current_chunk = 0
                consecutive += 1
            else:
                current_chunk += 1
                if consecutive > max_consecutive:
                    max_consecutive = consecutive
                consecutive = 0
        
        if current_chunk > 0:
            chunks.append(current_chunk)
        if consecutive > max_consecutive:
            max_consecutive = consecutive

        return {
            "blank_line_ratio": num_blank / total_lines,
            "avg_vertical_chunk_size": float(np.mean(chunks)) if chunks else total_lines,
            "max_consecutive_newlines": max_consecutive
        }
