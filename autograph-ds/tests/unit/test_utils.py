from src.utils import flatten_dna

def test_flatten_dna_simple():
    dna = {
        'a': 1,
        'b': {'c': 2, 'd': 3}
    }
    flat = flatten_dna(dna)
    assert flat['a'] == 1
    assert flat['b_c'] == 2
    assert flat['b_d'] == 3

def test_flatten_dna_special_handling():
    dna = {
        'node_type_counts': {
            'func': 0.1,
            'class': 0.2
        },
        'quote_preference': {
            "'": 5,
            '"': 10
        }
    }
    flat = flatten_dna(dna)
    assert flat['node_func'] == 0.1
    assert flat['node_class'] == 0.2
    assert flat['quote_single'] == 5
    assert flat['quote_double'] == 10

def test_flatten_dna_recursive():
    dna = {
        'x': {
            'y': {
                'z': 100
            }
        }
    }
    flat = flatten_dna(dna)
    assert flat['x_y_z'] == 100
