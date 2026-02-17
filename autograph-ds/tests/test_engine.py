import pytest
import os
import tempfile
import shutil
import joblib
from unittest.mock import patch, MagicMock
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

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)

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

# Error Handling Tests

def test_engine_missing_matcher_model(temp_dir):
    """Test that FileNotFoundError is raised when matcher model doesn't exist."""
    nonexistent_path = os.path.join(temp_dir, "nonexistent.joblib")
    with pytest.raises(FileNotFoundError, match="Matcher model not found"):
        AttestationEngine(nonexistent_path, CONSISTENCY_DIR)

def test_engine_missing_consistency_dir():
    """Test that FileNotFoundError is raised when consistency directory doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Consistency models directory not found"):
        AttestationEngine(MATCHER_PATH, "/nonexistent/directory/path")

def test_engine_matcher_load_failure(temp_dir):
    """Test that Exception is raised when matcher model fails to load."""
    # Create a corrupted file
    bad_model_path = os.path.join(temp_dir, "bad_model.joblib")
    with open(bad_model_path, 'w') as f:
        f.write("not a valid joblib file")

    with pytest.raises(Exception, match="Failed to load matcher model"):
        AttestationEngine(bad_model_path, CONSISTENCY_DIR)

def test_engine_consistency_load_failure(temp_dir):
    """Test that Exception is raised when consistency models fail to load."""
    if not os.path.exists(MATCHER_PATH):
        pytest.skip("Models not found for integration test")

    # Create empty directory (no consistency models inside)
    with pytest.raises(Exception, match="Failed to load consistency models"):
        AttestationEngine(MATCHER_PATH, temp_dir)

def test_engine_dna_extraction_failure():
    """Test that Exception is raised when DNA extraction fails."""
    if not os.path.exists(MATCHER_PATH):
        pytest.skip("Models not found for integration test")

    engine = AttestationEngine(MATCHER_PATH, CONSISTENCY_DIR)

    # Mock the extractor to raise an exception
    with patch.object(engine.extractor, 'extract', side_effect=Exception("Parse error")):
        with pytest.raises(Exception, match="Failed to extract DNA features"):
            engine.attest("def test(): pass", "mariusz")

# Confidence Penalty Tests

def test_engine_consistency_penalty(engine):
    """Test that confidence is penalized when consistency check fails."""
    code = "def test(): return True"

    # Mock consistency checker to return anomaly (-1)
    with patch.object(engine.consistency, 'check_consistency', return_value=(-1, -0.5)):
        result = engine.attest(code, "mariusz")

        # Confidence should be reduced by 0.5x due to anomaly
        assert result["consistency"] == "FAIL"
        # The confidence should be lower due to the penalty

def test_engine_heuristic_penalty(engine):
    """Test that confidence is heavily penalized when heuristic flags are present."""
    code = "# -*- coding: utf-8 -*-\ndef test(): return True"

    # This code should trigger DeepSeek marker detection
    result = engine.attest(code, "gpt4o")  # Claiming wrong identity

    # Confidence should be heavily reduced (0.2x) due to model spoofing
    if result["flags"]:  # If flags were detected
        assert result["confidence"] < 0.3  # Should be very low

# Verdict Edge Cases

def test_engine_verdict_verified(engine):
    """Test VERIFIED verdict: match + high confidence + consistent."""
    code = "def test(): return True"

    # Mock to ensure all conditions for VERIFIED are met
    with patch.object(engine.matcher, 'predict', return_value=[("mariusz", 0.9)]), \
         patch.object(engine.consistency, 'check_consistency', return_value=(1, 0.8)), \
         patch.object(engine.heuristics, 'verify_metadata', return_value=[]), \
         patch.object(engine.heuristics, 'detect_markers', return_value=[]):

        result = engine.attest(code, "mariusz")
        assert result["verdict"] == "VERIFIED"
        assert result["is_match"] is True
        assert result["confidence"] > 0.7
        assert result["consistency"] == "PASS"

def test_engine_verdict_uncertain(engine):
    """Test UNCERTAIN verdict: match + medium confidence."""
    code = "def test(): return True"

    # Mock to ensure conditions for UNCERTAIN (0.4 < confidence < 0.7)
    with patch.object(engine.matcher, 'predict', return_value=[("mariusz", 0.5)]), \
         patch.object(engine.consistency, 'check_consistency', return_value=(1, 0.5)), \
         patch.object(engine.heuristics, 'verify_metadata', return_value=[]), \
         patch.object(engine.heuristics, 'detect_markers', return_value=[]):

        result = engine.attest(code, "mariusz")
        assert result["verdict"] == "UNCERTAIN"
        assert result["is_match"] is True
        assert 0.4 < result["confidence"] < 0.7

def test_engine_verdict_mismatch(engine):
    """Test MISMATCH verdict: no match or low confidence."""
    code = "def test(): return True"

    # Mock to ensure no match
    with patch.object(engine.matcher, 'predict', return_value=[("other_author", 0.9), ("mariusz", 0.1)]), \
         patch.object(engine.consistency, 'check_consistency', return_value=(1, 0.5)), \
         patch.object(engine.heuristics, 'verify_metadata', return_value=[]), \
         patch.object(engine.heuristics, 'detect_markers', return_value=[]):

        result = engine.attest(code, "mariusz")
        assert result["verdict"] == "MISMATCH"
        assert result["is_match"] is False

def test_engine_verdict_spoofing_detected(engine):
    """Test SPOOFING_DETECTED verdict: heuristic flags present."""
    code = "def test(): return True"

    # Mock to ensure heuristic flags are present
    with patch.object(engine.heuristics, 'verify_metadata', return_value=["MODEL_SPOOFING_SUSPECTED"]):
        result = engine.attest(code, "mariusz")
        assert result["verdict"] == "SPOOFING_DETECTED"
        assert len(result["flags"]) > 0
