"""
Tests for Custom AI Identity Attestation.

Demonstrates using the attestation engine to verify that a custom AI model
(e.g., "MyBot 1.0") generates code within the bounds of its own training corpus.

Key concept: AI models can have their own identities, just like human authors.
The attestation engine verifies corpus consistency the same way for both.
The ai_probability field remains useful even for known AI identities because
it helps detect humans trying to spoof an AI identity.

Scenarios:
1. MyBot self-attestation - code matches its own corpus (VERIFIED)
2. Human spoofing MyBot - human code claimed as AI (MISMATCH)
3. Different AI as MyBot - GPT-4o code claimed as MyBot (MISMATCH)
4. MyBot corpus drift - MyBot's style has changed over time (UNCERTAIN)
"""
import pytest
from unittest.mock import patch
from src.engine import AttestationEngine
from src.models.base import IdentityModel, AnomalyModel


class MyBotMatcher(IdentityModel):
    def __init__(self, probs):
        super().__init__()
        self._probs = probs
        self.features = ["a", "b"]
        self._trained = True

    def train(self, X, y, feature_names):
        pass

    def predict_probs(self, X_df):
        return self._probs

    def save(self, path):
        pass

    def load(self, path):
        pass

    def is_trained(self):
        return True


class MyBotConsistency(AnomalyModel):
    def __init__(self, pred, score_val):
        super().__init__()
        self._pred = pred
        self._score = score_val
        self.features = ["a", "b"]
        self._trained = True

    def train(self, X, feature_names, identities):
        pass

    def score(self, identity, X_df):
        return self._pred, self._score

    def save(self, path):
        pass

    def load(self, path):
        pass

    def is_trained(self):
        return True


EXTENDED_AI_IDENTITIES = frozenset(['deepseek', 'gpt4o', 'gemini', 'mybot_v1'])


def test_custom_ai_self_attestation():
    """Scenario 1: MyBot 1.0 code verified against its own corpus. VERIFIED."""
    matcher = MyBotMatcher([("mybot_v1", 0.85), ("gpt4o", 0.10), ("mariusz", 0.05)])
    consistency = MyBotConsistency(pred=1, score_val=0.8)
    engine = AttestationEngine(matcher_impl=matcher, consistency_impl=consistency)

    with patch.object(AttestationEngine, 'AI_IDENTITIES', EXTENDED_AI_IDENTITIES):
        result = engine.attest("def generate(): return self.model.predict(x)", claimed_identity="mybot_v1")

    assert result["corpus_probability"] >= 0.80
    assert result["is_match"] is True
    assert result["ai_probability"] >= 0.90
    assert result["detected_ai_identity"] == "mybot_v1"
    assert result["consistency"] == "PASS"
    assert result["verdict"] == "VERIFIED"


def test_human_spoofing_custom_ai():
    """Scenario 2: Human code claimed as MyBot 1.0. MISMATCH."""
    matcher = MyBotMatcher([("mariusz", 0.70), ("raymond", 0.15), ("gpt4o", 0.10), ("mybot_v1", 0.05)])
    consistency = MyBotConsistency(pred=-1, score_val=-0.3)
    engine = AttestationEngine(matcher_impl=matcher, consistency_impl=consistency)

    with patch.object(AttestationEngine, 'AI_IDENTITIES', EXTENDED_AI_IDENTITIES):
        result = engine.attest("def process(data): return [x for x in data if x]", claimed_identity="mybot_v1")

    assert result["corpus_probability"] <= 0.10
    assert result["detected_identity"] == "mariusz"
    assert result["is_match"] is False
    assert result["ai_probability"] <= 0.20
    assert result["consistency"] == "FAIL"
    assert result["verdict"] == "MISMATCH"


def test_different_ai_claiming_custom_ai_identity():
    """Scenario 3: GPT-4o code claimed as MyBot 1.0. MISMATCH."""
    matcher = MyBotMatcher([("gpt4o", 0.80), ("mybot_v1", 0.10), ("gemini", 0.05), ("mariusz", 0.05)])
    consistency = MyBotConsistency(pred=-1, score_val=-0.2)
    engine = AttestationEngine(matcher_impl=matcher, consistency_impl=consistency)

    with patch.object(AttestationEngine, 'AI_IDENTITIES', EXTENDED_AI_IDENTITIES):
        result = engine.attest("def solve(problem): return optimal_solution(problem)", claimed_identity="mybot_v1")

    assert result["corpus_probability"] <= 0.15
    assert result["is_match"] is False
    assert result["ai_probability"] >= 0.90
    assert result["detected_ai_identity"] == "gpt4o"
    assert result["verdict"] == "MISMATCH"


def test_custom_ai_corpus_drift():
    """Scenario 4: MyBot 1.0 output drifted from training corpus."""
    matcher = MyBotMatcher([("mybot_v1", 0.50), ("gpt4o", 0.30), ("gemini", 0.15), ("mariusz", 0.05)])
    consistency = MyBotConsistency(pred=-1, score_val=-0.4)
    engine = AttestationEngine(matcher_impl=matcher, consistency_impl=consistency)

    with patch.object(AttestationEngine, 'AI_IDENTITIES', EXTENDED_AI_IDENTITIES):
        result = engine.attest("def transform(input): return modified(input)", claimed_identity="mybot_v1")

    assert result["corpus_probability"] == 0.50
    assert result["is_match"] is True
    assert result["ai_probability"] >= 0.90
    assert result["detected_ai_identity"] == "mybot_v1"
    assert result["confidence"] < 0.30
    assert result["consistency"] == "FAIL"
    # With new verdict logic: corpus_probability 0.50 >= 0.3 → UNCERTAIN
    # Consistency FAIL reduces confidence but doesn't change verdict from match
    assert result["verdict"] == "UNCERTAIN"
