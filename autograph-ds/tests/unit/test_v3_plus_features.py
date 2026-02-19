import pytest
from src.features.extractor import LogicalDNAExtractor

def test_syntactic_bias():
    extractor = LogicalDNAExtractor()
    code = """
def foo():
    items = list()
    if x is True:
        try:
            try:
                pass
            except:
                pass
        except:
            pass
"""
    dna = extractor.extract(code, enabled_buckets=["syntactic_bias"])
    assert dna["literal_collection_ratio"] == 0.0 # because we used list()
    assert dna["boolean_style_score"] == 1.0 # because of 'is True'
    assert dna["exception_depth"] > 1.0

def test_logic_flow():
    extractor = LogicalDNAExtractor()
    code = """
def foo(data):
    return list(map(lambda x: x * 2, data))
"""
    dna = extractor.extract(code, enabled_buckets=["logic_flow"])
    assert dna["functional_score"] == 1.0

def test_logic_flow_procedural():
    extractor = LogicalDNAExtractor()
    code = """
def foo(data):
    res = []
    for x in data:
        res.append(x * 2)
    return res
"""
    dna = extractor.extract(code, enabled_buckets=["logic_flow"])
    assert dna["functional_score"] == 0.0
