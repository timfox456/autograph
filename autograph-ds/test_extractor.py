from src.features.extractor import LogicalDNAExtractor
import json

code_sample = """
def calculate_total(items, discount=0.1):
    \"\"\"Calculates the total price with a discount.\"\"\"
    total = sum(item['price'] for item in items)
    if total > 100:
        return total * (1 - discount)
    return total

class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, name, price):
        self.items.append({"name": name, "price": price})
"""

extractor = LogicalDNAExtractor()
dna = extractor.extract(code_sample)

print(json.dumps(dna, indent=2))
