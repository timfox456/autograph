"""Model-based authorship verification analysis.

This module provides tools for extracting embeddings from pre-trained
code understanding models (CodeBERT, GraphCodeBERT, UniXcoder, CLAVE)
and evaluating their effectiveness for authorship attribution.
"""

from src.model_analysis.device import get_device, get_device_info
from src.model_analysis.cache_manager import EmbeddingCache
from src.model_analysis.embedders import HuggingFaceEmbedder, get_embedder
from src.model_analysis.clave_embedder import CLAVEEmbedder, get_clave_embedder

__all__ = [
    "get_device",
    "get_device_info",
    "EmbeddingCache",
    "HuggingFaceEmbedder",
    "get_embedder",
    "CLAVEEmbedder",
    "get_clave_embedder",
]
