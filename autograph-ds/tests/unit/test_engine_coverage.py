import pytest
from src.engine import AttestationEngine


def test_engine_implementation_fallback():
    class RawImpl:
        def predict_probs(self, X):
            return [("alice", 0.9)]

        def score(self, identity, X):
            return 1, 0.9

        def is_trained(self):
            return True

    class Wrapper:
        def __init__(self, impl):
            self.implementation = impl

        # Does NOT have predict_probs, predict, score, or check_consistency

    impl = RawImpl()
    matcher_wrapper = Wrapper(impl)
    consistency_wrapper = Wrapper(impl)

    engine = AttestationEngine(matcher_impl=matcher_wrapper, consistency_impl=consistency_wrapper)
    result = engine.attest("def foo(): pass", claimed_identity="alice")

    assert result["detected_identity"] == "alice"
    assert result["confidence"] == 0.9


def test_engine_legacy_features_fallback():
    class LegacyMatcher:
        features = ["a", "b"]

        def predict(self, X):
            return [("bob", 0.8)]

    class LegacyConsistency:
        features = ["a", "b"]

        def check_consistency(self, identity, X):
            return 1, 0.7

    engine = AttestationEngine(matcher_impl=LegacyMatcher(), consistency_impl=LegacyConsistency())
    result = engine.attest("def bar(): pass", claimed_identity="bob")

    assert result["detected_identity"] == "bob"
    assert result["confidence"] == 0.8


def test_engine_is_trained_exception_handling():
    class BrokenModel:
        def is_trained(self):
            raise Exception("Broken")

        features = ["a"]

        def predict(self, X):
            return [("charlie", 1.0)]

        def score(self, id, X):
            return 1, 1.0

    # This should hit the 'except Exception' and fall back to 'features' check
    engine = AttestationEngine(matcher_impl=BrokenModel(), consistency_impl=BrokenModel())
    result = engine.attest("pass", "charlie")
    assert result["detected_identity"] == "charlie"


def test_engine_missing_interface_methods():
    class EmptyModel:
        def is_trained(self):
            return True

    engine = AttestationEngine(matcher_impl=EmptyModel(), consistency_impl=EmptyModel())

    with pytest.raises(AttributeError, match="Matcher implementation must provide"):
        engine.attest("pass", "alice")


def test_engine_consistency_missing_interface_methods():
    class ValidMatcher:
        def is_trained(self):
            return True

        def predict_probs(self, X):
            return [("alice", 1.0)]

    class EmptyConsistency:
        def is_trained(self):
            return True

    engine = AttestationEngine(matcher_impl=ValidMatcher(), consistency_impl=EmptyConsistency())

    with pytest.raises(AttributeError, match="Consistency implementation must provide"):
        engine.attest("pass", "alice")


def test_engine_is_trained_implementation_exception_handling():
    class BrokenImpl:
        def is_trained(self):
            raise Exception("Broken")

    class Wrapper:
        def __init__(self, impl):
            self.implementation = impl

        features = ["a"]

        def predict(self, X):
            return [("alice", 1.0)]

        def score(self, id, X):
            return 1, 1.0

    # This should hit the 'except Exception' in the implementation check
    engine = AttestationEngine(
        matcher_impl=Wrapper(BrokenImpl()), consistency_impl=Wrapper(BrokenImpl())
    )
    result = engine.attest("pass", "alice")
    assert result["detected_identity"] == "alice"


def test_engine_heuristic_ai_leak():
    # Test heuristic penalty for AI leak in human-claimed code
    class MockMatcher:
        def is_trained(self):
            return True

        def predict_probs(self, X):
            return [("user_alice", 0.9)]

    class MockConsistency:
        def is_trained(self):
            return True

        def score(self, id, X):
            return 1, 1.0

    engine = AttestationEngine(matcher_impl=MockMatcher(), consistency_impl=MockConsistency())
    code = """
<thought>
I should write some python code.
</thought>
def foo(): pass
"""
    result = engine.attest(code, claimed_identity="user_alice")
    # Should have a massive penalty (0.2x)
    assert result["confidence"] < 0.2
    assert result["verdict"] == "SPOOFING_DETECTED"
    # The flags themselves are currently not returned in the result dict, but used for confidence scaling and verdict.


def test_extractor_empty_line_coverage():
    from src.features.extractor import LogicalDNAExtractor

    extractor = LogicalDNAExtractor()
    code = "x = 1\n\ny = 2"  # Includes empty line
    dna = extractor.extract(code, enabled_buckets=["micro_stylistics"])
    assert "indent_width" in dna
