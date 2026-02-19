import pytest
from src.engine import AttestationEngine
from src.models.base import IdentityModel, AnomalyModel


class DummyIdentityModel(IdentityModel):
    def __init__(self):
        super().__init__()
        self.features = ["a", "b"]
        self._trained = True

    def train(self, X, y, feature_names):
        pass

    def predict_probs(self, X_df):
        # Always strongly prefer 'mariusz'
        return [("mariusz", 0.9), ("other", 0.1)]

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def is_trained(self) -> bool:
        return True


class DummyAnomalyModel(AnomalyModel):
    def __init__(self):
        super().__init__()
        self.features = ["a", "b"]
        self._trained = True

    def train(self, X, feature_names, identities):
        pass

    def score(self, identity: str, X_df):
        # Always anomalous to trigger penalty
        return -1, -0.5

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def is_trained(self) -> bool:
        return True


def test_engine_uses_raw_impls_without_legacy_wrappers():
    engine = AttestationEngine(matcher_impl=DummyIdentityModel(), consistency_impl=DummyAnomalyModel())
    result = engine.attest("def t(): pass", claimed_identity="mariusz")

    # Should choose 'mariusz' and apply anomaly penalty (0.5x)
    assert result["detected_identity"] == "mariusz"
    assert 0.4 < result["confidence"] < 0.7  # 0.9 * 0.5 = 0.45
    assert result["consistency"] == "FAIL"


class UntrainedMatcher(IdentityModel):
    def train(self, X, y, feature_names):
        pass
    def predict_probs(self, X_df):
        return [("mariusz", 1.0)]
    def save(self, path: str):
        pass
    def load(self, path: str):
        pass
    def is_trained(self) -> bool:
        return False


class TrainedMatcher(DummyIdentityModel):
    pass


class UntrainedAnomaly(AnomalyModel):
    def train(self, X, feature_names, identities):
        pass
    def score(self, identity: str, X_df):
        return 1, 0.0
    def save(self, path: str):
        pass
    def load(self, path: str):
        pass
    def is_trained(self) -> bool:
        return False


def test_engine_blocks_untrained_matcher():
    engine = AttestationEngine(matcher_impl=UntrainedMatcher(), consistency_impl=DummyAnomalyModel())
    with pytest.raises(ValueError, match="Matcher model is not trained or loaded"):
        engine.attest("def t(): pass", claimed_identity="mariusz")


def test_engine_blocks_untrained_consistency():
    engine = AttestationEngine(matcher_impl=TrainedMatcher(), consistency_impl=UntrainedAnomaly())
    with pytest.raises(ValueError, match="Consistency model is not trained or loaded"):
        engine.attest("def t(): pass", claimed_identity="mariusz")


def test_engine_fallbacks_to_legacy_methods():
    class LegacyMatcher:
        features = ["a", "b"]
        def is_trained(self):
            return True
        def predict(self, feature_dict):
            # Legacy API: returns list of (label, prob)
            return [("mariusz", 0.8), ("other", 0.2)]

    class LegacyConsistency:
        features = ["a", "b"]
        def is_trained(self):
            return True
        def check_consistency(self, claimed_identity, feature_dict):
            # Legacy API: return (-1, score) to trigger penalty
            return -1, -0.1

    engine = AttestationEngine(matcher_impl=LegacyMatcher(), consistency_impl=LegacyConsistency())
    result = engine.attest("def x(): pass", claimed_identity="mariusz")

    # Should consume legacy methods and apply penalty => 0.8 * 0.5 = 0.4
    assert result["detected_identity"] == "mariusz"
    assert abs(result["confidence"] - 0.4) < 1e-6
    assert result["consistency"] == "FAIL"
