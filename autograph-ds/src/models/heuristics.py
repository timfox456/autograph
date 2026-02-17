import re

class HeuristicDetector:
    def __init__(self):
        # Patterns that are highly characteristic of specific models
        self.model_markers = {
            "claude35": [
                r"<thought>", # Sometimes leaks if prompt wasn't clean
                r"Here is the code to solve the problem:", # Stereotypical intro
                r"I have implemented the requested functionality"
            ],
            "gpt4o": [
                r"// Standard implementation", # Often uses very standard comments
                r"def solve\("
            ],
            "deepseek": [
                r"# -\*- coding: utf-8 -\*-", # Often includes encoding header
            ]
        }

    def detect_markers(self, code: str):
        findings = []
        for model, patterns in self.model_markers.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    findings.append({
                        "model": model,
                        "marker": pattern,
                        "confidence": 0.8 # Heuristic confidence
                    })
        return findings

    def verify_metadata(self, code: str, claimed_identity: str):
        """
        Returns flags if the code contradicts the claimed identity.
        """
        flags = []
        
        # Example: Human users usually don't have XML-like tags in their code unless specifically working on XML.
        if claimed_identity.startswith("user_"):
            if "<thought>" in code:
                flags.append("AI_LEAK_DETECTED")
                
        # Example: Detecting DeepSeek headers in something claimed as GPT
        if claimed_identity == "gpt4o" and "# -*- coding: utf-8 -*-" in code:
            flags.append("MODEL_SPOOFING_SUSPECTED")
            
        return flags
