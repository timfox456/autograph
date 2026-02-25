from src.features.extractor import LogicalDNAExtractor


def test_cfg_complexity_early_exits():
    extractor = LogicalDNAExtractor()
    code = """
def process(x):
    if x < 0:
        return 0
    if x == 0:
        raise ValueError("Zero")
    for i in range(x):
        if i > 10:
            continue
    return x
"""
    dna = extractor.extract(code, enabled_buckets=["cfg_complexity"])
    cfg = dna
    assert cfg["exit_density"] > 0
    assert cfg["break_statement_count"] == 0
    # Guard clauses (return 0 and raise ValueError) are in the first few lines
    assert cfg["guard_clause_score"] >= 2


def test_cfg_while_true():
    extractor = LogicalDNAExtractor()
    code = """
while True:
    if condition():
        break
"""
    dna = extractor.extract(code, enabled_buckets=["cfg_complexity"])
    assert dna["while_true_ratio"] == 1.0
    assert dna["break_statement_count"] == 1


def test_comment_stylistics():
    extractor = LogicalDNAExtractor()
    code = """
# INSTRUCTIONAL: Calculate the sum of two numbers.
# explanatory: this is done because we need the total.
def add(a, b):
    \"\"\"This is a docstring.\"\"\"
    # TODO: add validation
    # SHOUTING COMMENT
    return a + b # Inline comment
"""
    dna = extractor.extract(code, enabled_buckets=["comment_stylistics"])
    assert dna["instructional_ratio"] > 0
    assert dna["explanatory_ratio"] > 0
    assert dna["debt_markers_count"] == 1
    assert dna["inline_comment_ratio"] > 0
    assert dna["has_docstrings"] is True
    assert dna["all_caps_ratio"] > 0


def test_dead_code_detection():
    extractor = LogicalDNAExtractor()
    code = """
# def old_func():
#     pass
x = 1
# y = 2
"""
    dna = extractor.extract(code, enabled_buckets=["comment_stylistics"])
    assert dna["dead_code_density"] > 0


def test_ast_trigrams():
    extractor = LogicalDNAExtractor()
    code = "x = 1 + 2"
    dna = extractor.extract(code, enabled_buckets=["ast_trigrams"])
    trigrams = dna["top_trigrams"]
    assert len(trigrams) > 0
    # Check for some expected sequence, e.g., module -> expression_statement -> assignment
    # The actual string depends on tree-sitter node types
    assert any("assignment" in k for k in trigrams.keys())
