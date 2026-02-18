import pytest
from src.features.extractor import LogicalDNAExtractor

@pytest.fixture
def extractor():
    return LogicalDNAExtractor()

def test_extractor_complex_nesting(extractor):
    code = """
def level1():
    if True:
        for i in range(1):
            while False:
                try:
                    with open('f') as f:
                        if True:
                            pass
                except:
                    pass
"""
    dna = extractor.extract(code, enabled_buckets=["structural_topology"])
    # Nesting: level1(1) -> if(2) -> for(3) -> while(4) -> try(5) -> with(6) -> if(7)
    assert dna['max_nesting_depth'] >= 7

def test_extractor_idiomatic_diversity(extractor):
    code = """
import os
from math import sqrt

class MyClass:
    @staticmethod
    def calc(x):
        return [i**2 for i in range(x) if i % 2 == 0]

def main():
    data = {"a": 1, "b": 2}
    try:
        val = data.get("c", 0)
        async with aiohttp.ClientSession() as session:
            pass
    except Exception as e:
        print(f"Error: {e}")
"""
    dna = extractor.extract(code, enabled_buckets=["logical_idioms", "structural_topology"])
    
    assert dna['list_comprehension_count'] == 1
    assert dna['try_except_count'] == 1
    assert dna['class_definition_count'] == 1
    assert dna['f_string_ratio'] > 0

def test_extractor_quote_preferences(extractor):
    code1 = "x = 'single'"
    code2 = 'x = "double"'
    code3 = "x = 'single'; y = \"double\""
    
    dna1 = extractor.extract(code1, enabled_buckets=["micro_stylistics"])
    dna2 = extractor.extract(code2, enabled_buckets=["micro_stylistics"])
    dna3 = extractor.extract(code3, enabled_buckets=["micro_stylistics"])
    
    assert dna1['quote_preference']["'"] > 0
    assert dna1['quote_preference']['"'] == 0
    
    assert dna2['quote_preference']['"'] > 0
    assert dna2['quote_preference']["'"] == 0
    
    assert dna3['quote_preference']["'"] > 0
    assert dna3['quote_preference']['"'] > 0

def test_extractor_naming_conventions(extractor):
    code = """
def MyCamelCase():
    SNAKE_CASE_CONST = 1
    local_variable = 2
    anotherLocal = 3
"""
    dna = extractor.extract(code, enabled_buckets=["logical_idioms"])
    
    # CamelCase (PascalCase) starts with Uppercase
    # camelCase starts with lowercase
    # snake_case is all lowercase with underscores
    
    # MyCamelCase -> 1
    # SNAKE_CASE_CONST -> usually counted as snake if underscore, but depends on regex
    # local_variable -> snake
    # anotherLocal -> camel
    
    assert dna['camel_case_ratio'] > 0
    assert dna['snake_case_ratio'] > 0
