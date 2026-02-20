import pytest
from src.engine import AttestationEngine
import math

@pytest.fixture
def engine():
    """Provides a default AttestationEngine instance for tests."""
    return AttestationEngine()

def test_compute_corpus_probability(engine):
    match_probs = [('user_a', 0.8), ('user_b', 0.1), ('user_c', 0.1)]
    
    # Test for claimed identity present in match_probs
    prob = engine._compute_corpus_probability(match_probs, 'user_a')
    assert prob == pytest.approx(0.8 / (0.8 + 0.1 + 0.1))
    
    # Test for a different claimed identity
    prob_b = engine._compute_corpus_probability(match_probs, 'user_b')
    assert prob_b == pytest.approx(0.1 / (0.8 + 0.1 + 0.1))
    
    # Test for claimed identity not in match_probs
    prob_d = engine._compute_corpus_probability(match_probs, 'user_d')
    assert prob_d == 0.0

    # Test with no other probabilities
    match_probs_single = [('user_a', 0.9)]
    prob_single = engine._compute_corpus_probability(match_probs_single, 'user_a')
    assert prob_single == pytest.approx(1.0)

def test_compute_corpus_percentile(engine):
    # Test with a positive score
    score_high = engine._compute_corpus_percentile(0.5, 'user_a')
    assert score_high == pytest.approx(1.0 / (1.0 + math.exp(-0.5)))
    
    # Test with a negative score
    score_low = engine._compute_corpus_percentile(-0.5, 'user_a')
    assert score_low == pytest.approx(1.0 / (1.0 + math.exp(0.5)))
    
    # Test with a zero score
    score_zero = engine._compute_corpus_percentile(0, 'user_a')
    assert score_zero == 0.5
    
    # Test with None score
    score_none = engine._compute_corpus_percentile(None, 'user_a')
    assert score_none == 0.5

def test_compute_ai_probability(monkeypatch, engine):
    monkeypatch.setattr(AttestationEngine, 'AI_IDENTITIES', frozenset(['gpt4o', 'gemini']))

    # Test with AI identities present
    match_probs_with_ai = [('user_a', 0.6), ('gpt4o', 0.3), ('gemini', 0.1)]
    ai_prob, top_ai, top_ai_prob = engine._compute_ai_probability(match_probs_with_ai)
    assert ai_prob == pytest.approx(0.3 + 0.1)
    assert top_ai == 'gpt4o'
    assert top_ai_prob == pytest.approx(0.3)

    # Test with no AI identities
    match_probs_no_ai = [('user_a', 0.8), ('user_b', 0.2)]
    ai_prob_none, top_ai_none, top_ai_prob_none = engine._compute_ai_probability(match_probs_no_ai)
    assert ai_prob_none == 0.0
    assert top_ai_none is None
    assert top_ai_prob_none == 0.0

    # Test with only one AI identity
    match_probs_one_ai = [('user_a', 0.7), ('gemini', 0.3)]
    ai_prob_one, top_ai_one, top_ai_prob_one = engine._compute_ai_probability(match_probs_one_ai)
    assert ai_prob_one == pytest.approx(0.3)
    assert top_ai_one == 'gemini'
    assert top_ai_prob_one == pytest.approx(0.3)
