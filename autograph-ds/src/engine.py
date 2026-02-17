from src.features.extractor import LogicalDNAExtractor
from src.models.supervised import IdentityMatcher
from src.models.anomaly import ConsistencyChecker
from src.models.heuristics import HeuristicDetector
import numpy as np

class AttestationEngine:
    def __init__(self, matcher_path, consistency_dir):
        self.extractor = LogicalDNAExtractor()
        self.matcher = IdentityMatcher()
        self.matcher.load(matcher_path)
        self.consistency = ConsistencyChecker()
        self.consistency.load(consistency_dir)
        self.heuristics = HeuristicDetector()

    def attest(self, code, claimed_identity, enabled_buckets=None):
        """
        Performs a full attestation of the code against a claimed identity.
        """
        # 1. Extract DNA
        dna = self.extractor.extract(code, enabled_buckets)
        
        # Flatten for models (simplified flattening for the engine)
        # In a real system, we'd use the same process_dataset logic
        flat_dna = self._flatten_for_engine(dna)
        
        # 2. Supervised Match
        match_probs = self.matcher.predict(flat_dna)
        top_match, top_prob = match_probs[0]
        
        # 3. Consistency Check
        consistency_pred, consistency_score = self.consistency.check_consistency(claimed_identity, flat_dna)
        
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
        # This should match the logic in process_dataset.py
        # For the pilot, let's keep it simple or import the function
        from process_dataset import flatten_dna
        return flatten_dna(dna)
