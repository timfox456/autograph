"""
Utility functions for the Autograph Data Science project.
"""

def flatten_dna(dna, prefix=''):
    """
    Flattens nested DNA dictionary into a flat structure for ML models.

    Args:
        dna: Dictionary containing extracted DNA features
        prefix: Prefix for nested keys (used in recursion)

    Returns:
        Flattened dictionary with all values at top level
    """
    items = {}
    for k, v in dna.items():
        if isinstance(v, dict):
            # Special handling for node_type_counts which is a dictionary of type distribution
            if k == 'node_type_counts':
                for type_name, count in v.items():
                    items[f"node_{type_name}"] = count
            elif k == 'quote_preference':
                for quote_type, count in v.items():
                    # Sanitize quote type for column name
                    safe_name = quote_type.replace('"', 'double').replace("'", 'single')
                    items[f"quote_{safe_name}"] = count
            else:
                items.update(flatten_dna(v, prefix=f"{k}_"))
        else:
            items[k] = v
    return items
