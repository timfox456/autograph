# Abstract: The Autograph Protocol

Autograph is a zero-trust provenance framework designed to establish a
verifiable chain of custody for digital logic. In an ecosystem where human and
synthetic outputs are increasingly blended, Autograph moves beyond simple
detection to provide rigorous attribution: guaranteeing that an output is
situated correctly within its claimed corpus—be it a specific human artisan’s
historical body of work or a verified generative model’s output stream.

The protocol utilizes a dual-layer verification architecture to enforce this
provenance:

Probabilistic Pedigree Analysis: Autograph employs a hybrid engine of
supervised stylometric modeling and unsupervised anomaly detection to validate
corpus consistency. By analyzing the structural and lexical "DNA" of a commit,
the system verifies that code tagged as "Human" aligns with the unique logical
topology of that individual’s corpus, and code tagged as "AI-Generated" matches
the known architectural fingerprints and stochastic patterns of the specific
model cited.

Immutable Logic Ledger: These attestations are anchored to a decentralized,
content-addressable ledger. By representing code as deterministic Graph
Signatures, Autograph ensures that once a block of code is attributed to a
corpus, its provenance is immutable. This creates a "Firewall of Origin" that
prevents semantic laundering and protects the integrity of both human
intellectual property and AI-generated assets.

Integrated via a real-time LSP (Language Server Protocol), Autograph acts as a
forensic gatekeeper. It identifies "Corpus Mismatches"—such as licensed AI
output being laundered into a human stream or proprietary human logic being
misattributed to a model—before they are committed to the repository. The
result is a transparent, Zero-Trust environment where the value of code is
derived from its proven source.


Core Principles:

Corpus Situating: Proving code exists within the expected statistical bounds of its claimed creator.

Structural Topology: Using graph-based analysis to identify logic-flow consistency across versions.

Laundering Prevention: Identifying when a "poisoned prompt" or "style-mimic" is used to move code between corpora dishonestly.
