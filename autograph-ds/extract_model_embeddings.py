#!/usr/bin/env python3
"""Extract embeddings from pre-trained models for authorship verification.

This script loads raw Python code samples and extracts embeddings using:
- CodeBERT (microsoft/codebert-base)
- GraphCodeBERT (microsoft/graphcodebert-base)
- UniXcoder (microsoft/unixcoder-base)
- CLAVE (davidaf3/CLAVE - if available)

Embeddings are cached to avoid re-computation on subsequent runs.

Usage:
    python extract_model_embeddings.py [--force]

Options:
    --force    Re-extract embeddings even if cached versions exist
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model_analysis.device import get_device, get_device_info
from src.model_analysis.cache_manager import EmbeddingCache
from src.model_analysis.embedders import get_embedder
from src.model_analysis.clave_embedder import get_clave_embedder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RAW_DATA_DIR = Path("research/data/raw")
EMBEDDING_CACHE_DIR = Path("research/model_analysis/data/embeddings")
MIN_TOKEN_COUNT = 50  # Skip files with fewer than 50 tokens


def load_raw_samples() -> Tuple[List[str], List[str], List[str]]:
    """Load raw Python code samples and extract labels.
    
    Returns:
        Tuple of (code_strings, identities, filenames)
    """
    logger.info(f"Loading raw samples from {RAW_DATA_DIR}...")
    
    codes = []
    identities = []
    filenames = []
    
    if not RAW_DATA_DIR.exists():
        logger.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        return [], [], []
    
    for py_file in sorted(RAW_DATA_DIR.glob("*.py")):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Skip very short files
            if len(code) < MIN_TOKEN_COUNT:
                logger.debug(f"Skipping {py_file.name}: too short ({len(code)} chars)")
                continue
            
            # Parse identity from filename (e.g., "human_mariusz_0.py" -> "mariusz")
            # or "ai_claude_5.py" -> "claude"
            parts = py_file.stem.split('_')
            if len(parts) >= 2:
                if parts[0] == 'human':
                    identity = parts[1]
                elif parts[0] == 'ai':
                    # Handle multi-part AI names like "deepseek_v3"
                    identity = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                else:
                    identity = parts[0]
            else:
                identity = "unknown"
            
            codes.append(code)
            identities.append(identity)
            filenames.append(py_file.name)
            
        except Exception as e:
            logger.warning(f"Failed to load {py_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(codes)} samples")
    
    # Log identity distribution
    from collections import Counter
    identity_counts = Counter(identities)
    logger.info("Identity distribution:")
    for identity, count in sorted(identity_counts.items()):
        logger.info(f"  {identity}: {count}")
    
    return codes, identities, filenames


def save_metadata(codes: List[str], identities: List[str], filenames: List[str]):
    """Save sample metadata for downstream use.
    
    Args:
        codes: List of code strings
        identities: List of identity labels
        filenames: List of original filenames
    """
    metadata_path = EMBEDDING_CACHE_DIR / "samples_metadata.json"
    
    metadata = {
        "n_samples": len(codes),
        "identities": identities,
        "filenames": filenames,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved sample metadata to {metadata_path}")


def extract_embeddings_for_model(
    model_name: str,
    codes: List[str],
    cache: EmbeddingCache,
    force: bool = False
) -> bool:
    """Extract embeddings for a single model.
    
    Args:
        model_name: Name of the model ('codebert', 'graphcodebert', 'unixcoder', 'clave')
        codes: List of code strings to embed
        cache: EmbeddingCache instance
        force: If True, re-extract even if cached
    
    Returns:
        True if successful
    """
    # Check cache
    if not force and cache.exists(model_name):
        logger.info(f"Using cached embeddings for {model_name}")
        return True
    
    logger.info(f"Extracting embeddings with {model_name}...")
    
    try:
        # Get embedder
        if model_name == 'clave':
            embedder = get_clave_embedder()
            if not embedder.is_available():
                logger.warning(f"CLAVE not available, skipping")
                return False
            embeddings = embedder.embed_batch(codes, batch_size=16, show_progress=True)
        else:
            embedder = get_embedder(model_name)
            embeddings = embedder.embed_batch(codes, batch_size=16, show_progress=True)
        
        if embeddings is None:
            logger.error(f"Failed to extract embeddings for {model_name}")
            return False
        
        # Save to cache
        metadata = {
            "model_name": model_name,
            "n_samples": len(codes),
            "embedding_dim": embeddings.shape[1],
        }
        cache.save(model_name, embeddings, metadata)
        
        logger.info(f"Successfully extracted {model_name} embeddings: {embeddings.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract embeddings for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from pre-trained models for authorship verification"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract embeddings even if cached versions exist"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["codebert", "graphcodebert", "unixcoder", "clave", "all"],
        default=["all"],
        help="Which models to extract embeddings from (default: all)"
    )
    
    args = parser.parse_args()
    
    # Determine which models to process
    if "all" in args.models:
        models = ["codebert", "graphcodebert", "unixcoder", "clave"]
    else:
        models = args.models
    
    logger.info(f"Will extract embeddings for: {', '.join(models)}")
    
    # Setup device
    device = get_device()
    device_info = get_device_info(device)
    logger.info(f"Using device: {device_info['name']} ({device_info['type']})")
    
    # Load raw samples
    codes, identities, filenames = load_raw_samples()
    if not codes:
        logger.error("No samples loaded. Exiting.")
        return 1
    
    # Save metadata
    save_metadata(codes, identities, filenames)
    
    # Setup cache
    cache = EmbeddingCache(EMBEDDING_CACHE_DIR)
    
    # Extract embeddings for each model
    results = {}
    for model_name in models:
        success = extract_embeddings_for_model(model_name, codes, cache, force=args.force)
        results[model_name] = success
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Extraction Summary:")
    logger.info("="*60)
    for model_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {model_name:20s} {status}")
    
    successful = sum(results.values())
    logger.info(f"\nTotal: {successful}/{len(models)} models successful")
    
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
