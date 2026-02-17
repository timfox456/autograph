# Autograph Strategic Roadmap

Building on the foundation of the [Manifesto](../docs/MANIFESTO.md), this roadmap outlines the technical evolution of Autograph from a Data Science proof-of-concept to a global provenance protocol.

## Phase 1: Foundation (Current)
**Goal**: Establish the feasibility of "Probabilistic Attestation" using source code.

- [x] Define "Logical DNA" feature extraction for Python.
- [x] Build ensemble model (Supervised + Anomaly Detection).
- [x] Develop proof-of-concept Engine (`autograph-ds`).
- [x] Create initial human/AI training dataset (17 samples).
- [ ] Expand dataset to 100+ samples per identity.
- [ ] Support multi-language extraction (JS/TS, Rust).

## Phase 2: Anchoring & Persistence (Near-term)
**Goal**: Transition from ephemeral demos to a persistent, verifiable record of provenance.

- [ ] **Ledger Integration**: Implement a lightweight commitment scheme (e.g., Merkle Trees anchored to a public ledger).
- [ ] **CLI Tooling**: Build a `autograph` CLI for local artifact attestation and ledger querying.
- [ ] **Schema Standardization**: Define a formal JSON schema for Autograph Attestations.
- [ ] **Continuous Training**: Implement a feedback loop where verified artifacts automatically update the identity's "Logical DNA" profile.

## Phase 3: Identity & Integration (Mid-term)
**Goal**: Make Autograph a seamless part of the creative and developer workflow.

- [ ] **Identity Management System**: Support pseudonymous and public identity binding.
- [ ] **LSP Implementation**: Provide real-time "Corpus Consistency" checks within VS Code and other IDEs.
- [ ] **GitHub/GitLab Actions**: Automated attestation checks for Pull Requests and CI/CD pipelines.
- [ ] **Model Fingerprinting**: Specialized detection for major LLM versions (e.g., GPT-4o vs. Claude 3.5 Sonnet).

## Phase 4: Expansion Beyond Code (Long-term)
**Goal**: Apply the principles of "Creative Lineage" to other domains of digital creation.

- [ ] **Audio/Visual DNA**: Research feature extraction for music (MIDI/Waveform) and digital art (SVG/Compositional structure).
- [ ] **Zero-Knowledge Provenance**: Allow creators to prove their work is "theirs" (consistent with their lineage) without revealing their private source code.
- [ ] **Decentralized Reputation Markets**: Enable communities to weight "Autograph Verified" content in discovery algorithms.

---

## Technical Challenges & Research Areas

- **Evolution of Style**: How does the system handle a human creator's natural evolution over years?
- **Adversarial Mimicry**: Can a model be prompted to specifically break its "Logical DNA" fingerprint?
- **Computational Cost**: Optimizing real-time feature extraction for multi-million line repositories.
