# Autograph Data Science (autograph-ds)

This subproject implements the **Probabilistic Attestation** engine, the core analytical component of the Autograph framework. It uses machine learning and static analysis to extract "Logical DNA" from source code and verify its origin.

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

```bash
pip install -r requirements.txt
```

### Running the Demo

The demo showcases various scenarios, including verified matches, mismatches, spoofing detection, and privacy-preserving attestation.

```bash
python autograph-ds/demo_attestation.py
```

### Training the Models

To rebuild the models from the processed dataset:

```bash
python autograph-ds/train_models.py
```
