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
python demo_attestation.py
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

We separate unit tests (logic/bugs) from model benchmarks (accuracy/metrics).

```bash
# Run unit tests
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest tests/unit -v

# Run model performance benchmarks
pytest tests/benchmarks -v -s
```

## Dependency Injection (DI)

The attestation engine supports dependency injection so you can plug in custom models:

- `AttestationEngine(matcher_impl=..., consistency_impl=...)`
- `matcher_impl` should implement an identity matcher with:
  - `predict_probs(X_df_or_dict) -> list[(label, prob)]` for a single sample
  - `save(path)`, `load(path)`, and `is_trained() -> bool`
  - Fallback supported: if `predict_probs()` is missing, engine calls legacy `predict()`.
- `consistency_impl` should implement an anomaly/consistency model with:
  - `score(identity, X_df_or_dict) -> (prediction, score)` where `prediction` is `1` (normal), `-1` (anomaly), or `None` (unknown identity);
    `score` is a float or `None` if unknown.
  - `save(path)`, `load(path)`, and `is_trained() -> bool`
  - Fallback supported: if `score()` is missing, engine calls legacy `check_consistency()`.

Notes:
- Unknown identities for consistency models should return `(None, None)` to standardize engine behavior across backends.
- The engine now validates both models are trained via `is_trained()` before attestation, with a legacy fallback to checking `features` presence.

### Model Performance Metrics

To get a detailed report of the current model's accuracy on the processed dataset:

```bash
python report_metrics.py
```

## Dataset Details

The dataset currently contains ~490 samples across 7 distinct identities (AIs and Humans).

- **AI models**: GPT-4o, Claude 3.5, DeepSeek V3, Llama 3, Gemini
- **Human authors**: mariusz, raymond (and others from greenfield collection)

## License

AGPL-3.0-or-later. This subproject is part of the core-adjacent exploratory work and follows the AGPL licensing model. Future SDKs/client libraries and IDE integrations are intended to be Apache-2.0.
