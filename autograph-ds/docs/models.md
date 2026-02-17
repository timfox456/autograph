# Attestation Engine & Models

The `AttestationEngine` is the central component that combines multiple analytical approaches to provide a probabilistic verdict on the origin of a code sample.

## Layered Model Architecture

The engine uses a tiered approach to validation:

### 1. Identity Matcher (Supervised)
- **Model**: Random Forest Classifier.
- **Goal**: To predict the most likely author or AI model from a set of known identities.
- **Output**: A probability distribution over all known identities.
- **Training**: Trained on the `dataset.csv` generated from the `research/data/raw` samples.

### 2. Consistency Checker (Unsupervised)
- **Model**: Isolation Forest (one per identity).
- **Goal**: To determine if the sample is "in-family" or "out-of-family" for the *claimed* identity.
- **Output**: A consistency score and a binary classification (1 for consistent, -1 for anomaly).
- **Purpose**: Even if the Matcher thinks a sample is "Identity A," the Consistency Checker can flag it as an anomaly if it doesn't fit the tightly defined profile of Identity A.

### 3. Heuristic Detector (Rule-based)
- **Goal**: To catch obvious indicators or contradictions.
- **Checks**:
    - **Metadata Contradictions**: e.g., claiming to be "GPT-4" but the code contains "Created by DeepSeek".
    - **AI Markers**: Detection of common AI-generated headers or comment patterns.
- **Impact**: Heuristic failures often result in an immediate `SPOOFING_DETECTED` verdict or a significant confidence penalty.

## Attestation Process

1.  **Extraction**: The `LogicalDNAExtractor` generates a DNA profile based on enabled buckets.
2.  **Scoring**:
    - Matcher provides a `detected_identity` and a base `confidence`.
    - Consistency Checker provides a pass/fail on the `claimed_identity`.
    - Heuristics are checked for flags and markers.
3.  **Verdict Synthesis**:
    - **VERIFIED**: High confidence match + Consistency PASS + No flags.
    - **UNCERTAIN**: Low confidence match or Consistency FAIL.
    - **MISMATCH**: The Matcher's top prediction does not match the claim.
    - **SPOOFING_DETECTED**: Heuristic flags triggered.

## Information Density & Privacy

The `privacy_density` score reflects how much information was used for the attestation. A higher density (more buckets enabled) leads to higher confidence but lower privacy.
