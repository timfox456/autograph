import pytest
from src.features.extractor import LogicalDNAExtractor

def test_layout_rhythm():
    extractor = LogicalDNAExtractor()
    code = """def foo():
    pass

def bar():
    x = 1

    return x
"""
    dna = extractor.extract(code, enabled_buckets=["layout_rhythm"])
    assert dna["blank_line_ratio"] > 0
    assert dna["max_consecutive_newlines"] >= 1
    assert dna["avg_vertical_chunk_size"] > 0

def test_lexical_complexity():
    extractor = LogicalDNAExtractor()
    code = """def calculate_result(x, y):
    val = x + y
    return val
"""
    dna = extractor.extract(code, enabled_buckets=["lexical_complexity"])
    assert dna["avg_identifier_length"] > 0
    assert dna["short_identifier_ratio"] > 0 # x, y
    assert dna["identifier_entropy"] > 0

def test_lexical_complexity_ai_style():
    extractor = LogicalDNAExtractor()
    # "Textbook" AI style with long, descriptive names
    code = """def process_user_data_list(user_data_entries):
    processed_entries = []
    for entry in user_data_entries:
        processed_entries.append(entry)
    return processed_entries
"""
    dna = extractor.extract(code, enabled_buckets=["lexical_complexity"])
    assert dna["avg_identifier_length"] > 10
    assert dna["short_identifier_ratio"] == 0
