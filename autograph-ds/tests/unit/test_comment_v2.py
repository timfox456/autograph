import pytest
from src.features.extractor import LogicalDNAExtractor

def test_comment_emojis_and_colorful_language():
    extractor = LogicalDNAExtractor()
    code = """
# This is a cool function 🚀
# WTF is this hack?
# Just a normal comment.
def foo():
    pass
"""
    dna = extractor.extract(code, enabled_buckets=["comment_stylistics"])
    
    assert dna["emoji_density"] > 0
    assert dna["colorful_language_ratio"] > 0
    # 'WTF' and 'hack' are in the colorful list
    assert dna["colorful_language_ratio"] == pytest.approx(1/3)
