import pytest
import os
from src.engine import AttestationEngine

# Use real models for integration testing if they exist
MATCHER_PATH = "research/models/matcher.joblib"
CONSISTENCY_DIR = "research/models"

@pytest.fixture
def engine():
    # Only run if models are present
    if not os.path.exists(MATCHER_PATH):
        pytest.skip("Models not found for integration test")
    return AttestationEngine(MATCHER_PATH, CONSISTENCY_DIR)

def test_engine_attestation_flow(engine):
    code = "def test(): return True"
    result = engine.attest(code, "mariusz")
    
    assert "verdict" in result
    assert "confidence" in result
    assert "detected_identity" in result
    assert "consistency" in result

def test_engine_privacy_filtering(engine):
    code = "def test(): return True"
    # Test that it works even with limited buckets
    result = engine.attest(code, "mariusz", enabled_buckets=["structural_topology"])
    
    assert result["privacy_density"] < 1.0
    assert result["verdict"] in ["VERIFIED", "MISMATCH", "SPOOFING_DETECTED", "UNCERTAIN"]

def test_engine_invalid_input(engine):
    with pytest.raises(ValueError):
        engine.attest("", "mariusz")
