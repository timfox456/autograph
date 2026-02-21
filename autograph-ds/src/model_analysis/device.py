"""Device auto-detection for model-based authorship verification.

Supports MPS (Apple Silicon) -> CUDA (Nvidia) -> CPU fallback.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(smoke_test: bool = True) -> torch.device:
    """Auto-detect best available device with priority: MPS > CUDA > CPU.
    
    Args:
        smoke_test: If True, perform a quick inference test on MPS to check
                   for compatibility issues.
    
    Returns:
        torch.device: Best available device
    """
    # Priority 1: MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        if smoke_test:
            try:
                # Smoke test: simple tensor operation
                test_tensor = torch.randn(10, 10).to(device)
                result = torch.matmul(test_tensor, test_tensor)
                logger.info(f"MPS smoke test passed. Using device: {device}")
                return device
            except Exception as e:
                logger.warning(f"MPS smoke test failed: {e}. Falling back to CPU.")
                return torch.device("cpu")
        else:
            logger.info(f"Using MPS device: {device}")
            return device
    
    # Priority 2: CUDA (Nvidia)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {device}")
        return device
    
    # Priority 3: CPU (fallback)
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU device: {device}")
        return device


def get_device_info(device: torch.device) -> dict:
    """Get detailed information about the device.
    
    Args:
        device: The torch device
    
    Returns:
        dict with device information
    """
    info = {
        "device": str(device),
        "type": device.type,
    }
    
    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        info["memory_total"] = torch.cuda.get_device_properties(device).total_memory
        info["memory_allocated"] = torch.cuda.memory_allocated(device)
        info["memory_reserved"] = torch.cuda.memory_reserved(device)
    
    elif device.type == "mps":
        info["name"] = "Apple Silicon MPS"
        # MPS doesn't provide memory info through torch APIs
    
    else:
        info["name"] = "CPU"
        info["cores"] = torch.get_num_threads()
    
    return info
