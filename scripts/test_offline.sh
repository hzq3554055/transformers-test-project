#!/bin/bash

# Test Offline Script
# This script tests the transformers testing framework without requiring internet connection

set -e  # Exit on any error

echo "🧪 Testing Transformers Framework (Offline Mode)"
echo "=============================================="

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

# Test offline functionality
echo ""
echo "🔍 Testing offline functionality..."
python src/transformers_test/offline_test.py

echo ""
echo "🧪 Running basic tests..."
python -c "
import sys
sys.path.append('src')
from transformers_test import setup_logging, set_seed, get_device
from transformers_test.offline_test import OfflineModelTester

# Setup
logger = setup_logging(level='INFO')
set_seed(42)
device = get_device()

print(f'✅ Setup completed - Device: {device}')

# Test offline model
tester = OfflineModelTester()
result = tester.test_inference('Hello world!')
print(f'✅ Offline inference test passed')
print(f'   Predictions: {result[\"predictions\"]}')
print(f'   Predicted class: {result[\"predicted_class\"]}')

# Test benchmark
benchmark = tester.benchmark_performance('Test sentence', num_runs=3, warmup_runs=1)
print(f'✅ Benchmark test passed - Mean time: {benchmark[\"mean_time\"]:.4f}s')

print('✅ All offline tests passed!')
"

echo ""
echo "📊 Testing data processing..."
python -c "
import sys
sys.path.append('src')
from transformers_test.offline_test import create_sample_data

# Create sample data
data = create_sample_data(num_samples=3, text_length=20)
print(f'✅ Sample data created: {len(data)} samples')
for i, sample in enumerate(data):
    print(f'   Sample {i+1}: {sample[\"text\"][:30]}... (label: {sample[\"label\"]})')
"

echo ""
echo "🎉 All offline tests completed successfully!"
echo ""
echo "📋 Summary:"
echo "  ✅ Framework imports correctly"
echo "  ✅ Offline model testing works"
echo "  ✅ Performance benchmarking works"
echo "  ✅ Sample data generation works"
echo "  ✅ No internet connection required"
echo ""
echo "🚀 Your transformers testing framework is ready for offline development!"
echo ""
echo "Next steps:"
echo "1. Start Jupyter: jupyter notebook"
echo "2. Open notebooks/01_quick_start.ipynb"
echo "3. Use offline_test.py for development without internet"
echo "4. When internet is available, use the full framework with real models"
