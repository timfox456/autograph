"""CLAVE embedder for authorship verification.

CLAVE (Contrastive Learning for Authorship Verification with Encoder) is a
specialized model for code authorship. It has a custom tokenizer and model
architecture different from standard HuggingFace models.

Download URL: https://www.reflection.uniovi.es/bigcode/download/2024/CLAVE/
"""

import os
import sys
import json
import math
import torch
import numpy as np
import requests
import rarfile
import zipfile
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import logging

from src.model_analysis.device import get_device

logger = logging.getLogger(__name__)

CLAVE_URLS = {
    "model": "https://www.reflection.uniovi.es/bigcode/download/2024/CLAVE/model.rar",
    "tokenizer": "https://www.reflection.uniovi.es/bigcode/download/2024/CLAVE/tokenizer_data.zip",
}

DEFAULT_CACHE_DIR = Path("~/.cache/clave").expanduser()


class CLAVEEmbedder:
    """Embedder for CLAVE model.
    
    Note: CLAVE requires downloading model weights and has a custom
    tokenizer that needs to be loaded separately from HuggingFace.
    """
    
    def __init__(self, device: Optional[torch.device] = None, cache_dir: Path = DEFAULT_CACHE_DIR):
        """Initialize CLAVE embedder.
        
        Args:
            device: torch device (auto-detected if None)
            cache_dir: Directory to cache CLAVE model files
        """
        # CLAVE uses aten::_nested_tensor_from_mask which is not implemented on MPS.
        # Force CPU regardless of the requested device.
        if device is not None and device.type != "cpu":
            logger.info(f"CLAVE: overriding device {device} → cpu (MPS/CUDA not supported)")
        self.device = torch.device("cpu")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.max_length = 512
        # Default; will be inferred after model load
        self.embedding_dim = 512
        
        # Try to load CLAVE
        self._available = self._try_load()
    
    def _download_file(self, url: str, dest: Path, desc: str) -> bool:
        """Download a file with progress bar.
        
        Args:
            url: URL to download from
            dest: Destination path
            desc: Description for progress bar
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Downloading {desc}...")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            
            with open(dest, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Downloaded {desc} to {dest}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {desc}: {e}")
            return False
    
    def _setup_clave(self) -> bool:
        """Download and extract CLAVE model files.
        
        Returns:
            True if setup successful
        """
        model_rar = self.cache_dir / "model.rar"
        tokenizer_zip = self.cache_dir / "tokenizer_data.zip"
        
        # Download model if needed
        if not model_rar.exists():
            if not self._download_file(CLAVE_URLS["model"], model_rar, "CLAVE model"):
                return False
        
        # Download tokenizer if needed
        if not tokenizer_zip.exists():
            if not self._download_file(CLAVE_URLS["tokenizer"], tokenizer_zip, "CLAVE tokenizer"):
                return False
        
        # Extract model — the archive contains CLAVE.pt directly (no model/ subdir)
        model_pt_candidates = [
            self.cache_dir / "model" / "CLAVE.pt",
            self.cache_dir / "CLAVE.pt",
        ]
        already_extracted = any(p.exists() and p.stat().st_size > 0 for p in model_pt_candidates)
        if not already_extracted:
            try:
                logger.info("Extracting CLAVE model...")
                try:
                    with rarfile.RarFile(model_rar) as rf:
                        rf.extractall(path=self.cache_dir)
                except Exception as rar_e:
                    logger.warning(f"rarfile extraction failed ({rar_e}), trying bsdtar...")
                    import subprocess
                    result = subprocess.run(
                        ["bsdtar", "xf", str(model_rar), "-C", str(self.cache_dir)],
                        capture_output=True, text=True,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"bsdtar failed: {result.stderr}")
                logger.info("Extracted CLAVE model")
            except Exception as e:
                logger.error(f"Failed to extract CLAVE model: {e}")
                return False
        
        # Extract tokenizer
        tokenizer_dir = self.cache_dir / "tokenizer_data"
        if not tokenizer_dir.exists():
            try:
                logger.info("Extracting CLAVE tokenizer...")
                with zipfile.ZipFile(tokenizer_zip) as zf:
                    zf.extractall(path=self.cache_dir)
                logger.info("Extracted CLAVE tokenizer")
            except Exception as e:
                logger.error(f"Failed to extract CLAVE tokenizer: {e}")
                return False
        
        return True
    
    def _try_load(self) -> bool:
        """Try to load CLAVE model and tokenizer.
        
        Returns:
            True if successfully loaded
        """
        try:
            # Setup/download if needed
            if not self._setup_clave():
                logger.warning("CLAVE setup failed. Model will not be available.")
                return False
            
            # Add extracted directories to sys.path for module discovery
            sys.path.insert(0, str(self.cache_dir))
            extracted_model_dir = self.cache_dir / "model"
            extracted_tokenizer_dir = self.cache_dir / "tokenizer_data"
            if extracted_model_dir.exists():
                sys.path.insert(0, str(extracted_model_dir))
            if extracted_tokenizer_dir.exists():
                sys.path.insert(0, str(extracted_tokenizer_dir))
            # Also support a vendored CLAVE repo if present in the project
            clave_repo = Path(__file__).parent.parent.parent / "CLAVE"
            if clave_repo.exists():
                sys.path.insert(0, str(clave_repo))
                # modules live in src/ subdirectory of the CLAVE repo
                clave_src = clave_repo / "src"
                if clave_src.exists():
                    sys.path.insert(0, str(clave_src))
            
            # Try to import CLAVE modules
            try:
                from model import FineTunedModel
                from tokenizer import SpTokenizer
            except ImportError:
                logger.warning("Could not import CLAVE modules. CLAVE not available.")
                return False
            
            # Load tokenizer
            self.tokenizer = SpTokenizer()
            
            # Load model — architecture from eval.py in CLAVE repo
            self.model = FineTunedModel(
                ntoken=self.tokenizer.get_vocab_size(),
                d_model=512,
                d_out=512,
                nhead=8,
                d_hid=2048,
                nlayers=6,
                dropout=0.1,
                use_layer_norm=True,
            )

            # Load weights
            model_path = self.cache_dir / "model" / "CLAVE.pt"
            if not model_path.exists():
                # Archive extracts CLAVE.pt directly to cache root (no model/ subdir)
                model_path = self.cache_dir / "CLAVE.pt"

            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                state_dict = checkpoint["model_state_dict"]
                # Strip _orig_mod. prefix added by torch.compile
                state_dict = {
                    (k[10:] if k.startswith("_orig_mod.") else k): v
                    for k, v in state_dict.items()
                }
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = 512
                logger.info(f"Loaded CLAVE model on {self.device} (dim={self.embedding_dim})")
                return True
            else:
                logger.warning(f"CLAVE model weights not found at {model_path}")
                return False
        
        except Exception as e:
            logger.warning(f"Failed to load CLAVE: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if CLAVE model is available.
        
        Returns:
            True if model loaded successfully
        """
        return self._available and self.model is not None
    
    def _truncate_head_tail(self, tokens: List[int]) -> List[int]:
        """Apply head+tail truncation for long sequences.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            Truncated list of token IDs
        """
        if len(tokens) <= self.max_length:
            return tokens
        
        half = self.max_length // 2
        head = tokens[:half]
        tail = tokens[-half:]
        return head + tail
    
    def embed(self, code: str) -> Optional[np.ndarray]:
        """Embed a single code snippet.
        
        Args:
            code: Python source code string
        
        Returns:
            Embedding vector (shape: embedding_dim,) or None if unavailable
        """
        if not self.is_available():
            logger.warning("CLAVE not available, cannot embed")
            return None
        
        try:
            # Tokenize — SpTokenizer.tokenizes() encodes a string to token ids
            tokens = self.tokenizer.tokenizes(code)
            tokens = self._truncate_head_tail(tokens)

            # Convert to tensor
            input_tensor = torch.tensor([tokens]).to(self.device)

            # Get embeddings via forward pass (avg_pool → layer_norm → linear → ReLU)
            with torch.no_grad():
                embedding = self.model(input_tensor)

            return embedding.cpu().numpy().flatten()
        
        except Exception as e:
            logger.error(f"Failed to embed with CLAVE: {e}")
            return None
    
    def embed_batch(self, codes: List[str], batch_size: int = 16, show_progress: bool = True) -> Optional[np.ndarray]:
        """Embed a batch of code snippets.
        
        Args:
            codes: List of Python source code strings
            batch_size: Number of samples to process at once
            show_progress: Whether to show progress bar
        
        Returns:
            Embedding matrix (shape: n_samples, embedding_dim) or None if unavailable
        """
        if not self.is_available():
            logger.warning("CLAVE not available, cannot embed batch")
            return None
        
        embeddings = []
        
        iterator = range(0, len(codes), batch_size)
        if show_progress:
            total_batches = math.ceil(len(codes) / max(1, batch_size))
            iterator = tqdm(iterator, desc="Embedding with CLAVE", total=total_batches)
        
        for i in iterator:
            batch = codes[i:i + batch_size]
            batch_embeddings = []
            
            for code in batch:
                embedding = self.embed(code)
                if embedding is not None:
                    batch_embeddings.append(embedding)
                else:
                    # Return zeros if embedding failed
                    batch_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
            
            embeddings.append(np.vstack(batch_embeddings))
        
        return np.vstack(embeddings)
    
    def __del__(self):
        """Clean up model from memory when done."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def get_clave_embedder(device: Optional[torch.device] = None, cache_dir: Path = DEFAULT_CACHE_DIR):
    """Factory function to get CLAVE embedder.
    
    Args:
        device: Optional torch device
        cache_dir: Optional custom cache directory
    
    Returns:
        CLAVEEmbedder instance (may not be available)
    """
    return CLAVEEmbedder(device, cache_dir)
