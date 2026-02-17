"""
Basic tests for the Autograph Data Science components.
"""
import pytest
from src.features.extractor import LogicalDNAExtractor
from src.utils import flatten_dna


def test_extraction_basic():
    """Test that basic DNA extraction works."""
    code = "def hello():\n    print('world')"
    extractor = LogicalDNAExtractor()
    dna = extractor.extract(code)

    assert 'total_nodes' in dna
    assert dna['total_nodes'] > 0
    assert 'max_nesting_depth' in dna


def test_extraction_with_buckets():
    """Test extraction with specific feature buckets."""
    code = """
def greet(name):
    message = f"Hello, {name}!"
    return message
"""
    extractor = LogicalDNAExtractor()

    # Only structural features
    dna_structural = extractor.extract(code, enabled_buckets=["structural_topology"])
    assert 'total_nodes' in dna_structural
    assert 'indent_type' not in dna_structural  # Stylistic feature should be absent

    # Only stylistic features
    dna_stylistic = extractor.extract(code, enabled_buckets=["micro_stylistics"])
    assert 'indent_type' in dna_stylistic
    assert 'total_nodes' not in dna_stylistic  # Structural feature should be absent


def test_extraction_empty_code():
    """Test that empty code raises ValueError."""
    extractor = LogicalDNAExtractor()

    with pytest.raises(ValueError, match="Code cannot be empty"):
        extractor.extract("")

    with pytest.raises(ValueError, match="Code cannot be empty"):
        extractor.extract("   ")


def test_flatten_dna():
    """Test DNA flattening utility."""
    dna = {
        'total_nodes': 10,
        'node_type_counts': {
            'function_definition': 0.5,
            'expression_statement': 0.3
        },
        'quote_preference': {
            "'": 2,
            '"': 3
        }
    }

    flat = flatten_dna(dna)

    assert flat['total_nodes'] == 10
    assert 'node_function_definition' in flat
    assert flat['node_function_definition'] == 0.5
    assert 'quote_single' in flat
    assert flat['quote_single'] == 2
    assert 'quote_double' in flat
    assert flat['quote_double'] == 3


def test_extraction_f_strings():
    """Test detection of f-string usage."""
    code_with_fstring = 'message = f"Hello, {name}!"'
    code_with_format = 'message = "Hello, {}".format(name)'

    extractor = LogicalDNAExtractor()

    dna_fstring = extractor.extract(code_with_fstring, enabled_buckets=["logical_idioms"])
    dna_format = extractor.extract(code_with_format, enabled_buckets=["logical_idioms"])

    # f-string code should have higher f_string_ratio
    assert dna_fstring['f_string_ratio'] > dna_format['f_string_ratio']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
