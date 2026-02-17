# Autograph

Autograph is a zero-trust provenance framework designed to establish trust in digital artifacts through a combination of **probabilistic attestation** and **immutable ledgers**.

## Vision

In an era of generative AI, knowing the origin and authenticity of code and content is critical. Autograph provides a decentralized way to verify "who" created "what" without necessarily compromising the privacy of the creator.


## Documentation

The project documentation is organized into three levels: Vision, Strategy, and Implementation.

*   **Vision**: [The Autograph Manifesto](./docs/MANIFESTO.md) - The long-term strategic end-state.
*   **Strategy**: [System Architecture](./docs/ARCHITECTURE.md) & [Strategic Roadmap](./docs/ROADMAP.md) - How we build the protocol.
*   **Implementation**: 
    *   [autograph-ds](./autograph-ds/README.md) - Technical details of the Probabilistic Engine.
    *   [Logical DNA Specification](autograph-ds/docs/logical_dna.md) - Feature extraction methodology.
    *   [Models & Architecture](autograph-ds/docs/models.md) - Analytical engine details.
    *   [Abstract](docs/abstract.md) - Foundational philosophy and technical theory.

---

## The Autograph Protocol

Autograph is more than a tool; it is a **voluntary provenance protocol** designed to establish creative continuity in the age of infinite digital production.

### Core Components

1.  **Probabilistic Attestation Engine**: Analyzing "Logical DNA" to verify that an artifact is situated within its claimed creative corpus (Human, AI, or Hybrid).
2.  **Immutable Logic Ledger**: Anchoring attestations to a decentralized ledger to build a verifiable creative lineage over time.
3.  **Creative Identity Layer**: Allowing creators to manage their digital personas and accumulate "Earned Continuity" as a form of reputation.

---

## Project Structure

*   **[autograph-ds](./autograph-ds)**: The Data Science component. Implements the Logical DNA extraction and the Probabilistic Attestation engine. This project demonstrates how we can identify authors or AI models based on their coding patterns.
*   **[docs](./docs)**: Technical specifications, roadmap, and foundational documents.

---

## Quick Start (Data Science Engine)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/autograph.git
cd autograph

# Set up the data science component
cd autograph-ds
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Run the demo
python demo_attestation.py
```

### Quick Example

```python
from autograph_ds.src.engine import AttestationEngine

# Load pre-trained models
engine = AttestationEngine(
    matcher_path="autograph-ds/research/models/matcher.joblib",
    consistency_dir="autograph-ds/research/models"
)

# Verify a code sample
code = """
def greet(name):
    return f"Hello, {name}!"
"""

result = engine.attest(code, claimed_identity="gpt4o")
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Development

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Install development dependencies
uv pip install -r autograph-ds/requirements-dev.txt

# Run tests
cd autograph-ds
pytest test_basic.py -v

# Format code
black src/
ruff check src/
```

## Project Status

**This is a proof-of-concept.** The current implementation demonstrates the theoretical approach with a small dataset (17 samples). A production system would require:

- Thousands of samples per identity
- Multi-language support (currently Python only)
- Distributed ledger integration
- LSP (Language Server Protocol) integration for IDE support
- Continuous model updates as coding styles evolve

## License

GPLv3

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Copyright (c) 2026  by Timothy M Fox
. All Rights Reserved.
