"""
Transformers Test Framework

A comprehensive testing framework for Hugging Face Transformers models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import ModelTester
from .data import DataProcessor
from .training import ModelTrainer
from .utils import setup_logging, get_device, set_seed
from .offline_test import OfflineModelTester
from .proxy_utils import setup_huggingface_proxy, get_proxy_info
from .gpu_utils import get_gpu_info, get_optimal_device, get_memory_usage, print_gpu_status

# Auto-setup proxy on import
try:
    setup_huggingface_proxy()
except Exception:
    pass  # Continue even if proxy setup fails

__all__ = [
    "ModelTester",
    "DataProcessor", 
    "ModelTrainer",
    "OfflineModelTester",
    "setup_logging",
    "get_device",
    "set_seed",
    "setup_huggingface_proxy",
    "get_proxy_info",
    "get_gpu_info",
    "get_optimal_device",
    "get_memory_usage",
    "print_gpu_status",
    "__version__",
    "__author__",
    "__email__",
]
