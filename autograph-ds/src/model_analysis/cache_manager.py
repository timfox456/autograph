"""Embedding cache manager for model-based authorship verification.

Manages loading and saving of model embeddings to avoid re-computation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("research/model_analysis/data/embeddings")


class EmbeddingCache:
    """Manages caching of model embeddings to disk."""
    
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
    
    def _get_cache_path(self, model_name: str) -> Path:
        """Get the cache file path for a model.
        
        Args:
            model_name: Name of the model (e.g., 'codebert', 'clave')
        
        Returns:
            Path to the .npy cache file
        """
        return self.cache_dir / f"{model_name}_embeddings.npy"
    
    def exists(self, model_name: str) -> bool:
        """Check if embeddings are cached for a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            True if cache exists
        """
        cache_path = self._get_cache_path(model_name)
        return cache_path.exists()
    
    def load(self, model_name: str) -> Optional[np.ndarray]:
        """Load cached embeddings for a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            numpy array of embeddings (shape: n_samples, embedding_dim)
            or None if cache doesn't exist
        """
        cache_path = self._get_cache_path(model_name)
        if not cache_path.exists():
            logger.debug(f"No cache found for {model_name}")
            return None
        
        try:
            embeddings = np.load(cache_path)
            logger.info(f"Loaded cached embeddings for {model_name}: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load cache for {model_name}: {e}")
            return None
    
    def save(self, model_name: str, embeddings: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Save embeddings to cache.
        
        Args:
            model_name: Name of the model
            embeddings: numpy array of embeddings (shape: n_samples, embedding_dim)
            metadata: Optional dict with additional info (model version, timestamp, etc.)
        """
        cache_path = self._get_cache_path(model_name)
        
        try:
            np.save(cache_path, embeddings)
            logger.info(f"Saved embeddings for {model_name}: {embeddings.shape}")
            
            # Update metadata
            self._update_metadata(model_name, metadata)
        except Exception as e:
            logger.error(f"Failed to save cache for {model_name}: {e}")
            raise
    
    def _update_metadata(self, model_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Update metadata file with model information.
        
        Args:
            model_name: Name of the model
            metadata: Optional dict with model info
        """
        import datetime
        
        # Load existing metadata or create new
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    all_metadata = json.load(f)
            except Exception:
                all_metadata = {}
        else:
            all_metadata = {}
        
        # Update metadata for this model
        model_metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if metadata:
            model_metadata.update(metadata)
        
        all_metadata[model_name] = model_metadata
        
        # Save back
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(all_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    def get_metadata(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get metadata for a model or all models.
        
        Args:
            model_name: Name of the model, or None for all
        
        Returns:
            Metadata dict or None
        """
        if not self.metadata_file.exists():
            return None
        
        try:
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            if model_name:
                return all_metadata.get(model_name)
            return all_metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
    
    def clear(self, model_name: Optional[str] = None):
        """Clear cache for a model or all models.
        
        Args:
            model_name: Name of the model, or None to clear all
        """
        if model_name:
            cache_path = self._get_cache_path(model_name)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache for {model_name}")
        else:
            # Clear all .npy files
            for cache_file in self.cache_dir.glob("*_embeddings.npy"):
                cache_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            logger.info("Cleared all embedding caches")
    
    def list_cached_models(self) -> list:
        """List all models with cached embeddings.
        
        Returns:
            List of model names
        """
        models = []
        for cache_file in self.cache_dir.glob("*_embeddings.npy"):
            model_name = cache_file.stem.replace("_embeddings", "")
            models.append(model_name)
        return sorted(models)
