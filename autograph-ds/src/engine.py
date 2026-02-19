from src.features.extractor import LogicalDNAExtractor
from src.models.supervised import IdentityMatcher
from src.models.anomaly import ConsistencyChecker
from src.models.heuristics import HeuristicDetector
from src.utils import flatten_dna
import numpy as np
import os

class AttestationEngine:
    def __init__(self, matcher_path=None, consistency_dir=None, matcher_impl=None, consistency_impl=None):
        """
        Initialize the AttestationEngine with models.

        Args:
            matcher_path: Path to the supervised matcher model file
            consistency_dir: Directory containing consistency checker models
            matcher_impl: Optional IdentityModel implementation
            consistency_impl: Optional AnomalyModel implementation
        """
        self.extractor = LogicalDNAExtractor()
        self.heuristics = HeuristicDetector()
        
        # Use provided implementations or default to legacy-compatible wrappers
        self.matcher = matcher_impl if matcher_impl else IdentityMatcher()
        self.consistency = consistency_impl if consistency_impl else ConsistencyChecker()

        if matcher_path:
            if not os.path.exists(matcher_path):
                raise FileNotFoundError(f"Matcher model not found at: {matcher_path}")
            try:
                self.matcher.load(matcher_path)
            except Exception as e:
                raise Exception(f"Failed to load matcher model: {e}")
        
        if consistency_dir:
            if not os.path.exists(consistency_dir):
                raise FileNotFoundError(f"Consistency models directory not found at: {consistency_dir}")
            try:
                self.consistency.load(consistency_dir)
            except Exception as e:
                raise Exception(f"Failed to load consistency models: {e}")

    def attest(self, code, claimed_identity, enabled_buckets=None):
        """
        Performs a full attestation of the code against a claimed identity.

        Args:
            code: Source code string to analyze
            claimed_identity: The identity claiming authorship
            enabled_buckets: List of feature buckets to use (default: all)

        Returns:
            Dictionary with attestation results and verdict

        Raises:
            ValueError: If code is empty or invalid
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")

        # Ensure matcher is trained/loaded
        matcher_trained = None
        if hasattr(self.matcher, 'is_trained'):
            try:
                matcher_trained = bool(self.matcher.is_trained())
            except Exception:
                matcher_trained = None
        if matcher_trained is None and hasattr(self.matcher, 'implementation') and hasattr(self.matcher.implementation, 'is_trained'):
            try:
                matcher_trained = bool(self.matcher.implementation.is_trained())
            except Exception:
                matcher_trained = None
        if matcher_trained is None:
            # Fallback to fragile check for legacy objects
            matcher_trained = hasattr(self.matcher, 'features') and bool(getattr(self.matcher, 'features'))

        if not matcher_trained:
            raise ValueError("Matcher model is not trained or loaded.")

        # Ensure consistency checker is trained/loaded
        consistency_trained = None
        if hasattr(self.consistency, 'is_trained'):
            try:
                consistency_trained = bool(self.consistency.is_trained())
            except Exception:
                consistency_trained = None
        if consistency_trained is None and hasattr(self.consistency, 'implementation') and hasattr(self.consistency.implementation, 'is_trained'):
            try:
                consistency_trained = bool(self.consistency.implementation.is_trained())
            except Exception:
                consistency_trained = None
        if consistency_trained is None:
            # Fallback: legacy objects expose features after training
            consistency_trained = hasattr(self.consistency, 'features') and bool(getattr(self.consistency, 'features'))
        if not consistency_trained:
            raise ValueError("Consistency model is not trained or loaded.")

        # 1. Extract DNA
        try:
            dna = self.extractor.extract(code, enabled_buckets)
        except Exception as e:
            raise Exception(f"Failed to extract DNA features: {e}")
        
        # Flatten for models (simplified flattening for the engine)
        # In a real system, we'd use the same process_dataset logic
        flat_dna = self._flatten_for_engine(dna)
        
        # 2. Supervised Match (support both interface and legacy wrapper)
        if hasattr(self.matcher, 'predict_probs'):
            match_probs = self.matcher.predict_probs(flat_dna)
        elif hasattr(self.matcher, 'predict'):
            match_probs = self.matcher.predict(flat_dna)
        elif hasattr(self.matcher, 'implementation') and hasattr(self.matcher.implementation, 'predict_probs'):
            match_probs = self.matcher.implementation.predict_probs(flat_dna)
        else:
            raise AttributeError("Matcher implementation must provide predict_probs() or predict().")
        top_match, top_prob = match_probs[0]
        
        # 3. Consistency Check (support both interface and legacy wrapper)
        if hasattr(self.consistency, 'score'):
            consistency_pred, consistency_score = self.consistency.score(claimed_identity, flat_dna)
        elif hasattr(self.consistency, 'check_consistency'):
            consistency_pred, consistency_score = self.consistency.check_consistency(claimed_identity, flat_dna)
        elif hasattr(self.consistency, 'implementation') and hasattr(self.consistency.implementation, 'score'):
            consistency_pred, consistency_score = self.consistency.implementation.score(claimed_identity, flat_dna)
        else:
            raise AttributeError("Consistency implementation must provide score() or check_consistency().")
        
        # 4. Heuristic Flags
        h_flags = self.heuristics.verify_metadata(code, claimed_identity)
        h_markers = self.heuristics.detect_markers(code)
        
        # 5. Combine results
        is_match = (top_match == claimed_identity)
        confidence = top_prob
        
        # Scale confidence based on consistency and heuristics
        if consistency_pred == -1:
            confidence *= 0.5 # Significant penalty for anomaly
            
        if h_flags:
            confidence *= 0.2 # Massive penalty for contradictory markers
            
        # Information density calculation
        bucket_count = len(enabled_buckets) if enabled_buckets else 3
        density_score = bucket_count / 3.0
        
        return {
            "claimed_identity": claimed_identity,
            "detected_identity": top_match,
            "is_match": is_match,
            "confidence": float(confidence),
            "consistency": "PASS" if consistency_pred == 1 else "FAIL" if consistency_pred == -1 else "UNKNOWN",
            "consistency_score": float(consistency_score) if consistency_score is not None else 0.0,
            "flags": h_flags,
            "markers": h_markers,
            "privacy_density": density_score,
            "verdict": self._get_verdict(is_match, confidence, consistency_pred, h_flags)
        }

    def _get_verdict(self, is_match, confidence, consistency, flags):
        if flags:
            return "SPOOFING_DETECTED"
        if is_match and confidence > 0.7 and consistency == 1:
            return "VERIFIED"
        if is_match and confidence > 0.4:
            return "UNCERTAIN"
        return "MISMATCH"

    def _flatten_for_engine(self, dna):
        """
        Flatten DNA dictionary for model consumption.
        Uses shared utility function to ensure consistency with training.
        """
        return flatten_dna(dna)
