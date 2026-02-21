"""HuggingFace model embedders for CodeBERT, GraphCodeBERT, and UniXcoder.

These models all use the HuggingFace transformers library and share a common
embedding extraction interface with mean pooling.
"""

import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
import logging
from tqdm import tqdm

from src.model_analysis.device import get_device

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "codebert": {
        "model_id": "microsoft/codebert-base",
        "max_length": 512,
        "embedding_dim": 768,
    },
    "graphcodebert": {
        "model_id": "microsoft/graphcodebert-base",
        "max_length": 512,
        "embedding_dim": 768,
    },
    "unixcoder": {
        "model_id": "microsoft/unixcoder-base",
        "max_length": 512,
        "embedding_dim": 768,
    },
}


class HuggingFaceEmbedder:
    """Base embedder for HuggingFace transformer models."""
    
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        """Initialize embedder for a HuggingFace model.
        
        Args:
            model_name: One of 'codebert', 'graphcodebert', 'unixcoder'
            device: torch device (auto-detected if None)
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Choose from: {list(MODEL_CONFIGS.keys())}")
        
        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.device = device or get_device()
        
        self.model_id = self.config["model_id"]
        self.max_length = self.config["max_length"]
        self.embedding_dim = self.config["embedding_dim"]
        
        # Load model and tokenizer
        logger.info(f"Loading {model_name} from {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded {model_name} on {self.device}")
    
    def _truncate_head_tail(self, tokens: List[int]) -> List[int]:
        """Apply head+tail truncation for long sequences.
        
        For sequences > max_length, take first half and last half.
        
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
    
    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        """Apply mean pooling over token embeddings.
        
        Args:
            model_output: Transformers model output
            attention_mask: Attention mask from tokenizer
        
        Returns:
            Mean-pooled embeddings (shape: batch_size, embedding_dim)
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def embed(self, code: str) -> np.ndarray:
        """Embed a single code snippet.
        
        Args:
            code: Python source code string
        
        Returns:
            Embedding vector (shape: embedding_dim,)
        """
        # Tokenize with truncation
        tokens = self.tokenizer.encode(code, add_special_tokens=True)
        tokens = self._truncate_head_tail(tokens)
        
        # Convert back to tensor
        inputs = {
            "input_ids": torch.tensor([tokens]).to(self.device),
            "attention_mask": torch.tensor([[1] * len(tokens)]).to(self.device),
        }
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = self._mean_pooling(outputs, inputs["attention_mask"])
        
        return embedding.cpu().numpy().flatten()
    
    def embed_batch(self, codes: List[str], batch_size: int = 16, show_progress: bool = True) -> np.ndarray:
        """Embed a batch of code snippets.
        
        Args:
            codes: List of Python source code strings
            batch_size: Number of samples to process at once
            show_progress: Whether to show progress bar
        
        Returns:
            Embedding matrix (shape: n_samples, embedding_dim)
        """
        embeddings = []
        
        iterator = range(0, len(codes), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Embedding with {self.model_name}", 
                          total=len(codes) // batch_size + 1)
        
        for i in iterator:
            batch = codes[i:i + batch_size]
            batch_embeddings = []
            
            for code in batch:
                # Tokenize each sample individually for head+tail truncation
                tokens = self.tokenizer.encode(code, add_special_tokens=True)
                tokens = self._truncate_head_tail(tokens)
                batch_embeddings.append(tokens)
            
            # Pad to max length in batch
            max_len = max(len(t) for t in batch_embeddings)
            padded_tokens = []
            attention_masks = []
            
            for tokens in batch_embeddings:
                padding = [self.tokenizer.pad_token_id] * (max_len - len(tokens))
                padded_tokens.append(tokens + padding)
                attention_masks.append([1] * len(tokens) + [0] * len(padding))
            
            # Convert to tensors
            inputs = {
                "input_ids": torch.tensor(padded_tokens).to(self.device),
                "attention_mask": torch.tensor(attention_masks).to(self.device),
            }
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embedding = self._mean_pooling(outputs, inputs["attention_mask"])
                embeddings.append(batch_embedding.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def __del__(self):
        """Clean up model from memory when done."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def get_embedder(model_name: str, device: Optional[torch.device] = None):
    """Factory function to get appropriate embedder.
    
    Args:
        model_name: One of 'codebert', 'graphcodebert', 'unixcoder'
        device: Optional torch device
    
    Returns:
        HuggingFaceEmbedder instance
    """
    return HuggingFaceEmbedder(model_name, device)
