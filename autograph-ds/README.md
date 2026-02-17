# Autograph Data Science (autograph-ds)

This subproject implements the **Probabilistic Attestation** engine, the core analytical component of the Autograph framework. It uses machine learning and static analysis to extract "Logical DNA" from source code and verify its origin.

> **Note**: This is a **proof-of-concept/demonstration** built on a limited training dataset (17 samples). It demonstrates the theoretical approach and technical feasibility but is not production-ready. A real-world system would require thousands of samples per identity for reliable attribution.

## Core Concepts

### Logical DNA
We extract features from source code across three distinct "buckets," allowing for a trade-off between attribution accuracy and developer privacy. See the [Logical DNA Specification](./docs/logical_dna.md) for more details.

## System Architecture

The attestation engine utilizes a multi-layered approach. Detailed information can be found in the [Attestation Engine & Models Specification](./docs/models.md).

## Project Structure

- `src/features/`: Code for AST parsing (using `tree-sitter`) and feature extraction.
- `src/models/`: Implementation of the three model types (Supervised, Anomaly, Heuristics).
- `src/engine.py`: The `AttestationEngine` that orchestrates extraction and scoring.
- `research/data/`: Raw and processed datasets used for training.
- `research/models/`: Serialized model files.

## Getting Started

### Installation

**Using uv (recommended):**
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
cd autograph-ds
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Install development dependencies (optional)
uv pip install -r requirements-dev.txt
```

**Using pip:**
```bash
cd autograph-ds
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Quick Start Example

```python
from src.engine import AttestationEngine

# Initialize the engine with trained models
engine = AttestationEngine(
    matcher_path="research/models/matcher.joblib",
    consistency_dir="research/models"
)

# Attest a code sample
code = """
def calculate_sum(numbers):
    return sum(numbers)
"""

result = engine.attest(code, claimed_identity="mariusz")

print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Detected Identity: {result['detected_identity']}")
```

### Running the Demo

The demo showcases various scenarios, including verified matches, mismatches, spoofing detection, and privacy-preserving attestation.

```bash
python autograph-ds/demo_attestation.py
```

### Training the Models

To rebuild the models from the processed dataset:

```bash
# First, process raw samples into a training dataset
python process_dataset.py

# Then train the models
python train_models.py
```

### Running Tests

```bash
# Run basic tests
pytest test_basic.py -v

# Run with coverage
pytest test_basic.py --cov=src --cov-report=html
```

## Development Workflow

1. **Collect training data**: Use `download_samples.py` or `collect_greenfield.py`
2. **Process dataset**: Run `process_dataset.py` to extract features
3. **Train models**: Run `train_models.py` to build the ML models
4. **Test attestation**: Use `demo_attestation.py` or the quick start example above

## Dataset Limitations

The current dataset consists of:
- **4 AI models**: GPT-4o, Claude 3.5, DeepSeek V3, Llama 3
- **5 human authors**: mariusz, kenneth, django, requests, fastapi
- **Total: 17 samples**

This is sufficient for demonstrating the concept but far too small for production use. A robust system would need:
- 100+ samples per identity
- More diverse code types (not just Python)
- Temporal validation (code from different time periods)
- Cross-validation to prevent overfitting
