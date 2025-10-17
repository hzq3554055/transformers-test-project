"""
GPU utilities for transformers testing framework.
"""

import torch
import logging
from typing import Dict, List, Optional, Any
import psutil
import time

logger = logging.getLogger(__name__)


def get_gpu_info() -> Dict[str, Any]:
    """
    Get comprehensive GPU information.
    
    Returns:
        Dictionary with GPU information
    """
    gpu_info = {
        "available": torch.cuda.is_available(),
        "count": 0,
        "current_device": None,
        "devices": []
    }
    
    if torch.cuda.is_available():
        gpu_info["count"] = torch.cuda.device_count()
        gpu_info["current_device"] = torch.cuda.current_device()
        
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": device_props.name,
                "total_memory_gb": device_props.total_memory / (1024**3),
                "major": device_props.major,
                "minor": device_props.minor,
                "multi_processor_count": device_props.multi_processor_count,
                "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                "memory_free_gb": (device_props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3)
            }
            gpu_info["devices"].append(device_info)
    
    return gpu_info


def get_optimal_device() -> str:
    """
    Get the optimal device for computation.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage in GB
    """
    memory_info = {}
    
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    memory_info["cpu_total_gb"] = cpu_memory.total / (1024**3)
    memory_info["cpu_available_gb"] = cpu_memory.available / (1024**3)
    memory_info["cpu_used_gb"] = cpu_memory.used / (1024**3)
    memory_info["cpu_percent"] = cpu_memory.percent
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            memory_info[f"gpu_{i}_total_gb"] = device_props.total_memory / (1024**3)
            memory_info[f"gpu_{i}_allocated_gb"] = torch.cuda.memory_allocated(i) / (1024**3)
            memory_info[f"gpu_{i}_reserved_gb"] = torch.cuda.memory_reserved(i) / (1024**3)
            memory_info[f"gpu_{i}_free_gb"] = (device_props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3)
    
    return memory_info


def benchmark_gpu_performance(
    model: torch.nn.Module,
    input_shape: tuple = (1, 512),
    num_runs: int = 10,
    warmup_runs: int = 3
) -> Dict[str, float]:
    """
    Benchmark GPU performance for a model.
    
    Args:
        model: PyTorch model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Performance metrics
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for GPU benchmarking")
    
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Benchmark runs
    torch.cuda.synchronize()
    times = []
    
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        "runs": num_runs
    }


def compare_cpu_gpu_performance(
    model: torch.nn.Module,
    input_shape: tuple = (1, 512),
    num_runs: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Compare CPU vs GPU performance.
    
    Args:
        model: PyTorch model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        
    Returns:
        Performance comparison results
    """
    results = {}
    
    # CPU benchmark
    model_cpu = model.cpu()
    model_cpu.eval()
    dummy_input_cpu = torch.randn(input_shape)
    
    cpu_times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model_cpu(dummy_input_cpu)
        end_time = time.time()
        cpu_times.append(end_time - start_time)
    
    results["cpu"] = {
        "mean_time": sum(cpu_times) / len(cpu_times),
        "min_time": min(cpu_times),
        "max_time": max(cpu_times),
        "std_time": (sum((t - sum(cpu_times) / len(cpu_times)) ** 2 for t in cpu_times) / len(cpu_times)) ** 0.5
    }
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        try:
            gpu_results = benchmark_gpu_performance(model, input_shape, num_runs, warmup_runs=2)
            results["gpu"] = gpu_results
            
            # Calculate speedup
            speedup = results["cpu"]["mean_time"] / results["gpu"]["mean_time"]
            results["speedup"] = speedup
            
        except Exception as e:
            logger.warning(f"GPU benchmark failed: {e}")
            results["gpu"] = None
            results["speedup"] = None
    else:
        results["gpu"] = None
        results["speedup"] = None
    
    return results


def print_gpu_status():
    """Print current GPU status."""
    gpu_info = get_gpu_info()
    
    print("🖥️  GPU Status:")
    print(f"  Available: {gpu_info['available']}")
    
    if gpu_info['available']:
        print(f"  Count: {gpu_info['count']}")
        print(f"  Current device: {gpu_info['current_device']}")
        
        for device in gpu_info['devices']:
            print(f"  GPU {device['id']}: {device['name']}")
            print(f"    Total memory: {device['total_memory_gb']:.1f} GB")
            print(f"    Allocated: {device['memory_allocated_gb']:.1f} GB")
            print(f"    Reserved: {device['memory_reserved_gb']:.1f} GB")
            print(f"    Free: {device['memory_free_gb']:.1f} GB")
    else:
        print("  No GPU available")


def optimize_for_gpu(model: torch.nn.Module, device: str = "cuda") -> torch.nn.Module:
    """
    Optimize model for GPU inference.
    
    Args:
        model: PyTorch model
        device: Target device
        
    Returns:
        Optimized model
    """
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
        model.eval()
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        logger.info(f"Model optimized for GPU: {torch.cuda.get_device_name()}")
    
    return model


if __name__ == "__main__":
    # Test GPU utilities
    print("🧪 Testing GPU Utilities")
    print("=" * 30)
    
    # Print GPU status
    print_gpu_status()
    
    # Get memory usage
    memory_info = get_memory_usage()
    print(f"\n💾 Memory Usage:")
    print(f"  CPU: {memory_info['cpu_used_gb']:.1f}GB / {memory_info['cpu_total_gb']:.1f}GB ({memory_info['cpu_percent']:.1f}%)")
    
    if torch.cuda.is_available():
        print(f"  GPU: {memory_info['gpu_0_allocated_gb']:.1f}GB / {memory_info['gpu_0_total_gb']:.1f}GB")
    
    # Test device selection
    optimal_device = get_optimal_device()
    print(f"\n🎯 Optimal device: {optimal_device}")
    
    print("\n✅ GPU utilities test completed!")
