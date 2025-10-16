"""
Utility functions for transformers testing.
"""

import os
import logging
import random
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import yaml
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={level}")
    return logger


def get_device(device: Optional[str] = None) -> str:
    """
    Get the best available device.
    
    Args:
        device: Preferred device
        
    Returns:
        Device string
    """
    if device:
        return device
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def create_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = f"experiment_{timestamp}"
    
    experiment_dir = Path(base_dir) / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return str(experiment_dir)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": model_size_mb
    }


def count_tokens(text: str, tokenizer) -> int:
    """
    Count tokens in text.
    
    Args:
        text: Input text
        tokenizer: Tokenizer to use
        
    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text))


def batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    """
    Split texts into batches.
    
    Args:
        texts: List of texts
        batch_size: Batch size
        
    Returns:
        List of text batches
    """
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


def ensure_dir(path: str):
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with memory info
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "allocated_memory_gb": torch.cuda.memory_allocated(0) / (1024**3),
        "cached_memory_gb": torch.cuda.memory_reserved(0) / (1024**3),
        "free_memory_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**3)
    }


def print_model_summary(model: torch.nn.Module):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    size_info = get_model_size(model)
    print(f"Model size: {size_info['model_size_mb']:.2f} MB")
    
    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        print(f"GPU memory: {gpu_info['allocated_memory_gb']:.2f}GB / {gpu_info['total_memory_gb']:.2f}GB")


def create_sample_texts(num_samples: int = 10, text_length: int = 100) -> List[str]:
    """
    Create sample texts for testing.
    
    Args:
        num_samples: Number of samples to create
        text_length: Length of each text
        
    Returns:
        List of sample texts
    """
    import random
    import string
    
    def generate_text(length):
        return ''.join(random.choices(string.ascii_letters + ' ', k=length))
    
    return [generate_text(text_length) for _ in range(num_samples)]
