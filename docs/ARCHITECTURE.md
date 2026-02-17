# Autograph System Architecture

This document defines the technical architecture of the Autograph protocol, transitioning from the initial Data Science proof-of-concept to a robust, multi-layered provenance system.

## The Autograph Protocol Stack

Autograph is organized into three primary layers, ensuring a separation of concerns between analytical attribution, cryptographic commitment, and identity management.

### 1. Attestation Layer (The Engine)
*Currently being explored in `autograph-ds`*

This layer is responsible for the probabilistic analysis of artifacts.
- **Logical DNA Extraction**: Using AST-based analysis (tree-sitter) to extract structural, lexical, and stylistic features.
- **Ensemble Scoring**: Combining supervised models (attribution), anomaly detection (consistency), and heuristics (linting/patterns).
- **Verdict Generation**: Producing a confidence-weighted attestation of "situatedness" within a claimed corpus.

### 2. Commitment Layer (The Ledger)
*Future Phase*

The commitment layer anchors attestations to ensure they are tamper-proof and verifiable over time.
- **Artifact Hashing**: Content-addressable identifiers for every piece of code or media.
- **Cryptographic Signatures**: Binding attestations to a specific version of the engine and the claimant's key.
- **Immutable Anchoring**: Committing attestation roots to a decentralized ledger (e.g., a DAG or Merkle tree or transparency log) to provide "Longitudinal Identity."

### 3. Identity Layer (The Persona)
*Future Phase*

This layer manages the creative identities that the system tracks.
- **Identity Types**:
    - **Human**: Verified through longitudinal consistency of style.
    - **AI**: Verified against model-specific fingerprints (e.g., LLM-specific stochastic patterns).
    - **Hybrid**: Recognizing collaborative creation (human-guided AI).
- **Privacy Controls**: Enabling pseudonymous participation while maintaining "Continuity of Signal."

---

## Data Flow: Artifact to Anchor

1. **Ingestion**: A digital artifact (e.g., a git commit) is submitted for attestation.
2. **Extraction**: The engine extracts "Logical DNA" across three buckets:
    - *Bucket A (Metadata)*: Committer info, timestamps, commit messages.
    - *Bucket B (Structure)*: AST topology, control flow, logic density.
    - *Bucket C (Stylistics)*: Naming conventions, indentation, lexical choices.
3. **Inference**: The engine compares the extracted DNA against the claimed identity's historical corpus.
4. **Attestation**: A signed claim is generated, containing:
    - The artifact hash.
    - The claimed identity.
    - The probabilistic verdict (Confidence, Consistency Score).
    - The engine version.
5. **Anchoring**: The attestation is committed to the ledger, contributing to the identity's "Creative Lineage."

## Strategic Technical Priorities

1. **Scaling to Large Corpora**: Moving from per-identity models to an efficient, shared embedding space for "Logical DNA."
2. **Cross-Language Support**: Expanding the `tree-sitter` based extractors to support JavaScript/TypeScript, Rust, and Go.
3. **LSP Integration**: Providing real-time provenance feedback within the developer's IDE.
4. **Zero-Knowledge Proofs**: Investigating ZK-attestations to prove "Situatedness" without revealing the underlying "Logical DNA" features.
