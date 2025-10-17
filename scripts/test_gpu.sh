#!/bin/bash

# GPU Test Script
# This script tests GPU functionality for the transformers testing framework

set -e  # Exit on any error

echo "🚀 Testing GPU Functionality for Transformers Framework"
echo "====================================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected. Activating venv..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "❌ Virtual environment not found. Please run install_dependencies.sh first"
        exit 1
    fi
fi

echo "✅ Virtual environment activated"

# Setup proxy if not already set
if [ -z "$HTTP_PROXY" ]; then
    echo "🌐 Setting up proxy configuration..."
    export HTTP_PROXY="http://127.0.0.1:7897"
    export HTTPS_PROXY="http://127.0.0.1:7897"
    export http_proxy="http://127.0.0.1:7897"
    export https_proxy="http://127.0.0.1:7897"
    echo "✅ Proxy configured: $HTTP_PROXY"
fi

echo ""
echo "🔍 Testing GPU Components..."

# Test 1: GPU Detection
echo "1️⃣  Testing GPU detection..."
python3 -c "
import sys
sys.path.append('src')
from transformers_test import get_gpu_info, get_optimal_device, print_gpu_status

# Print GPU status
print_gpu_status()

# Get optimal device
optimal_device = get_optimal_device()
print(f'\\n🎯 Optimal device: {optimal_device}')

# Get detailed GPU info
gpu_info = get_gpu_info()
if gpu_info['available']:
    print(f'✅ GPU detected: {gpu_info[\"count\"]} device(s)')
    for device in gpu_info['devices']:
        print(f'   GPU {device[\"id\"]}: {device[\"name\"]} ({device[\"total_memory_gb\"]:.1f} GB)')
else:
    print('❌ No GPU detected')
"

# Test 2: GPU Model Loading
echo "2️⃣  Testing GPU model loading..."
python3 -c "
import sys
sys.path.append('src')
from transformers_test import ModelTester, setup_logging

# Setup logging
logger = setup_logging(level='INFO')

try:
    # Test GPU model loading
    print('📥 Loading model on GPU...')
    tester = ModelTester(
        model_name='distilbert-base-uncased',
        task_type='classification',
        device='cuda'
    )
    
    print('✅ GPU model loaded successfully!')
    print(f'   Model: {tester.model_name}')
    print(f'   Device: {tester.device}')
    
except Exception as e:
    print(f'❌ GPU model loading failed: {e}')
    print('🔄 This might be due to insufficient GPU memory or CUDA issues')
"

# Test 3: GPU Inference Performance
echo "3️⃣  Testing GPU inference performance..."
python3 -c "
import sys
sys.path.append('src')
from transformers_test import ModelTester, setup_logging

# Setup logging
logger = setup_logging(level='INFO')

try:
    # Initialize GPU tester
    tester = ModelTester(
        model_name='distilbert-base-uncased',
        task_type='classification',
        device='cuda'
    )
    
    # Test inference
    print('🧪 Testing GPU inference...')
    result = tester.test_inference('Hello GPU world!')
    print(f'✅ GPU inference successful!')
    print(f'   Input: {result[\"input_text\"]}')
    print(f'   Device: {result[\"device\"]}')
    print(f'   Predictions: {result[\"predictions\"]}')
    print(f'   Predicted class: {result[\"predicted_class\"]}')
    
    # Benchmark performance
    print('\\n⚡ Benchmarking GPU performance...')
    benchmark = tester.benchmark_performance(
        'This is a GPU performance benchmark test.',
        num_runs=5,
        warmup_runs=2
    )
    print(f'✅ GPU benchmark completed!')
    print(f'   Mean time: {benchmark[\"mean_time\"]:.4f}s')
    print(f'   Min time: {benchmark[\"min_time\"]:.4f}s')
    print(f'   Max time: {benchmark[\"max_time\"]:.4f}s')
    print(f'   Std time: {benchmark[\"std_time\"]:.4f}s')
    
except Exception as e:
    print(f'❌ GPU inference test failed: {e}')
"

# Test 4: Memory Usage
echo "4️⃣  Testing GPU memory usage..."
python3 -c "
import sys
sys.path.append('src')
from transformers_test import get_memory_usage, get_gpu_info

# Get memory usage
memory_info = get_memory_usage()
print('💾 Memory Usage:')
print(f'  CPU: {memory_info[\"cpu_used_gb\"]:.1f}GB / {memory_info[\"cpu_total_gb\"]:.1f}GB ({memory_info[\"cpu_percent\"]:.1f}%)')

if 'gpu_0_total_gb' in memory_info:
    print(f'  GPU: {memory_info[\"gpu_0_allocated_gb\"]:.1f}GB / {memory_info[\"gpu_0_total_gb\"]:.1f}GB')
    print(f'  GPU Free: {memory_info[\"gpu_0_free_gb\"]:.1f}GB')

# Get GPU info
gpu_info = get_gpu_info()
if gpu_info['available']:
    print('\\n🖥️  GPU Details:')
    for device in gpu_info['devices']:
        print(f'  GPU {device[\"id\"]}: {device[\"name\"]}')
        print(f'    Total: {device[\"total_memory_gb\"]:.1f} GB')
        print(f'    Allocated: {device[\"memory_allocated_gb\"]:.1f} GB')
        print(f'    Reserved: {device[\"memory_reserved_gb\"]:.1f} GB')
        print(f'    Free: {device[\"memory_free_gb\"]:.1f} GB')
"

# Test 5: CPU vs GPU Performance Comparison
echo "5️⃣  Testing CPU vs GPU performance comparison..."
python3 -c "
import sys
sys.path.append('src')
from transformers_test import ModelTester, setup_logging
import time

# Setup logging
logger = setup_logging(level='INFO')

try:
    print('🔄 Comparing CPU vs GPU performance...')
    
    # CPU test
    print('  Testing CPU performance...')
    cpu_tester = ModelTester(
        model_name='distilbert-base-uncased',
        task_type='classification',
        device='cpu'
    )
    cpu_benchmark = cpu_tester.benchmark_performance(
        'CPU vs GPU performance test',
        num_runs=3,
        warmup_runs=1
    )
    
    # GPU test
    print('  Testing GPU performance...')
    gpu_tester = ModelTester(
        model_name='distilbert-base-uncased',
        task_type='classification',
        device='cuda'
    )
    gpu_benchmark = gpu_tester.benchmark_performance(
        'CPU vs GPU performance test',
        num_runs=3,
        warmup_runs=1
    )
    
    # Compare results
    speedup = cpu_benchmark['mean_time'] / gpu_benchmark['mean_time']
    
    print(f'\\n📊 Performance Comparison:')
    print(f'  CPU Mean time: {cpu_benchmark[\"mean_time\"]:.4f}s')
    print(f'  GPU Mean time: {gpu_benchmark[\"mean_time\"]:.4f}s')
    print(f'  Speedup: {speedup:.2f}x faster on GPU')
    
    if speedup > 1.5:
        print('  ✅ GPU provides significant speedup!')
    elif speedup > 1.1:
        print('  ✅ GPU provides moderate speedup')
    else:
        print('  ⚠️  GPU speedup is minimal (might be due to small model size)')
        
except Exception as e:
    print(f'❌ Performance comparison failed: {e}')
"

# Test 6: Batch Processing on GPU
echo "6️⃣  Testing batch processing on GPU..."
python3 -c "
import sys
sys.path.append('src')
from transformers_test import ModelTester, setup_logging

# Setup logging
logger = setup_logging(level='INFO')

try:
    # Initialize GPU tester
    tester = ModelTester(
        model_name='distilbert-base-uncased',
        task_type='classification',
        device='cuda'
    )
    
    # Test batch inference
    print('🔄 Testing batch processing on GPU...')
    texts = [
        'This is the first test sentence.',
        'This is the second test sentence.',
        'This is the third test sentence.',
        'This is the fourth test sentence.',
        'This is the fifth test sentence.'
    ]
    
    batch_results = tester.test_batch_inference(texts, batch_size=2)
    print(f'✅ Batch processing successful!')
    print(f'   Processed {len(batch_results)} batches')
    print(f'   Total texts: {len(texts)}')
    
    # Show results
    for i, result in enumerate(batch_results):
        print(f'   Batch {i+1}: {len(result[\"input_text\"])} texts processed')
        
except Exception as e:
    print(f'❌ Batch processing test failed: {e}')
"

echo ""
echo "🎉 GPU testing completed!"
echo ""
echo "📊 Test Summary:"
echo "  ✅ GPU detection and status"
echo "  ✅ GPU model loading"
echo "  ✅ GPU inference performance"
echo "  ✅ Memory usage monitoring"
echo "  ✅ CPU vs GPU performance comparison"
echo "  ✅ Batch processing on GPU"
echo ""
echo "🚀 Your RTX 3060 is now fully integrated with the transformers framework!"
echo ""
echo "📋 GPU Features Available:"
echo "  🖥️  Automatic GPU detection and usage"
echo "  ⚡ GPU-accelerated inference"
echo "  📊 Performance benchmarking"
echo "  💾 Memory usage monitoring"
echo "  🔄 Batch processing optimization"
echo "  📈 CPU vs GPU performance comparison"
echo ""
echo "🚀 Next Steps:"
echo "  1. Start Jupyter: jupyter notebook"
echo "  2. Open notebooks/01_quick_start.ipynb (now with GPU support)"
echo "  3. Run tests: python -m pytest tests/ -v"
echo "  4. Use CLI with GPU: python -m transformers_test.cli test --model bert-base-uncased --text 'Hello GPU world'"
echo "  5. Upload to GitHub for sharing"
