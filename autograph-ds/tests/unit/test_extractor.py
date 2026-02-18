import pytest
from src.features.extractor import LogicalDNAExtractor
from src.features.buckets import ALL_BUCKETS

@pytest.fixture
def extractor():
    return LogicalDNAExtractor()

def test_extraction_basic(extractor):
    code = "def hello():\n    print('world')"
    dna = extractor.extract(code)
    assert 'total_nodes' in dna
    assert dna['total_nodes'] > 0
    assert 'max_nesting_depth' in dna

def test_structural_features(extractor):
    code = """
def outer():
    def inner():
        if True:
            for i in range(10):
                pass
"""
    dna = extractor.extract(code, enabled_buckets=["structural_topology"])
    # 4 levels of nesting: outer -> inner -> if -> for
    assert dna['max_nesting_depth'] >= 4
    assert 'node_type_counts' in dna

def test_stylistic_features(extractor):
    code = "def test():\n\tprint(\"hello\")\n\treturn 'world'"
    dna = extractor.extract(code, enabled_buckets=["micro_stylistics"])
    
    # Tabs used
    assert dna['indent_type'] == 1.0
    # Quotes used
    assert dna['quote_preference']['"'] >= 1
    assert dna['quote_preference']["'"] >= 1

def test_idiomatic_features(extractor):
    code = """
def camelCaseFunction():
    snake_case_var = [x for x in range(10)]
    message = f"Value: {snake_case_var}"
    return message
"""
    dna = extractor.extract(code, enabled_buckets=["logical_idioms"])
    
    assert dna['snake_case_ratio'] > 0
    assert dna['camel_case_ratio'] > 0
    assert dna['f_string_ratio'] == 1.0
    assert dna['list_comprehension_count'] == 1

def test_empty_code(extractor):
    with pytest.raises(ValueError, match="Code cannot be empty"):
        extractor.extract("")

def test_syntax_error_handling(extractor):
    # tree-sitter is resilient, but let's see how it handles garbage
    code = "if: while for"
    dna = extractor.extract(code)
    # Should still return some features, even if node_type_counts includes ERROR
    assert 'total_nodes' in dna
    assert dna['total_nodes'] > 0
