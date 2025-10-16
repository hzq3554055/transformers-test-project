#!/bin/bash

# Benchmark Models Script
# This script benchmarks different transformer models for performance comparison

set -e  # Exit on any error

echo "⚡ Benchmarking Transformer Models"
echo "=================================="

# Default parameters
MODELS="bert-base-uncased,distilbert-base-uncased,roberta-base"
TEXT="This is a benchmark test sentence for evaluating model performance."
RUNS=10
WARMUP=3
OUTPUT_DIR="./results/benchmarks"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --text)
            TEXT="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models MODELS    Comma-separated list of models to benchmark"
            echo "  --text TEXT        Text to use for benchmarking"
            echo "  --runs RUNS        Number of benchmark runs (default: 10)"
            echo "  --warmup WARMUP     Number of warmup runs (default: 3)"
            echo "  --output DIR       Output directory for results (default: ./results/benchmarks)"
            echo "  --help             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --models 'bert-base-uncased,distilbert-base-uncased' --runs 20"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📋 Benchmark Configuration:"
echo "  Models: $MODELS"
echo "  Text: $TEXT"
echo "  Runs: $RUNS"
echo "  Warmup: $WARMUP"
echo "  Output: $OUTPUT_DIR"
echo ""

# Convert comma-separated models to array
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

# Initialize results
RESULTS_FILE="$OUTPUT_DIR/benchmark_results.json"
SUMMARY_FILE="$OUTPUT_DIR/benchmark_summary.txt"

echo "🚀 Starting benchmark..."

# Create Python script for benchmarking
cat > /tmp/benchmark_script.py << EOF
import sys
import json
import time
from pathlib import Path
sys.path.append('src')

from transformers_test import ModelTester, setup_logging

def benchmark_model(model_name, text, runs, warmup):
    """Benchmark a single model."""
    try:
        print(f"  Loading {model_name}...")
        tester = ModelTester(model_name=model_name, task_type='classification')
        
        print(f"  Running benchmark ({runs} runs, {warmup} warmup)...")
        start_time = time.time()
        benchmark = tester.benchmark_performance(text, num_runs=runs, warmup_runs=warmup)
        end_time = time.time()
        
        # Get model info
        model_info = tester.get_model_info()
        
        result = {
            'model_name': model_name,
            'benchmark_time': end_time - start_time,
            'mean_time': benchmark['mean_time'],
            'min_time': benchmark['min_time'],
            'max_time': benchmark['max_time'],
            'std_time': benchmark['std_time'],
            'runs': benchmark['runs'],
            'num_parameters': model_info['num_parameters'],
            'model_size_mb': model_info['model_size_mb'],
            'device': model_info['device']
        }
        
        print(f"    Mean time: {benchmark['mean_time']:.4f}s")
        print(f"    Parameters: {model_info['num_parameters']:,}")
        print(f"    Size: {model_info['model_size_mb']:.2f} MB")
        
        return result
        
    except Exception as e:
        print(f"    Error: {e}")
        return {
            'model_name': model_name,
            'error': str(e)
        }

# Setup logging
logger = setup_logging(level='INFO')

# Benchmark all models
results = []
for model_name in sys.argv[1:]:
    print(f"\\n🔍 Benchmarking {model_name}...")
    result = benchmark_model(model_name, "$TEXT", $RUNS, $WARMUP)
    results.append(result)

# Save results
Path("$RESULTS_FILE").parent.mkdir(parents=True, exist_ok=True)
with open("$RESULTS_FILE", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\\n📊 Results saved to $RESULTS_FILE")

# Print summary
print("\\n📈 Benchmark Summary:")
print("=" * 50)
for result in results:
    if 'error' not in result:
        print(f"{result['model_name']:20} | {result['mean_time']:8.4f}s | {result['num_parameters']:10,} params | {result['model_size_mb']:6.2f} MB")
    else:
        print(f"{result['model_name']:20} | ERROR: {result['error']}")

EOF

# Run the benchmark script
python3 /tmp/benchmark_script.py "${MODEL_ARRAY[@]}"

# Clean up
rm -f /tmp/benchmark_script.py

echo ""
echo "✅ Benchmark completed!"
echo ""
echo "Results saved to:"
echo "  - Detailed results: $RESULTS_FILE"
echo "  - Summary: $SUMMARY_FILE"
echo ""
echo "To view results:"
echo "  cat $RESULTS_FILE | jq ."
echo "  python3 -c \"import json; print(json.load(open('$RESULTS_FILE')))\""
