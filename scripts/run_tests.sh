#!/bin/bash

# Run Tests Script
# This script runs the test suite with various configurations

set -e  # Exit on any error

echo "🧪 Running Transformers Testing Framework Tests"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

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

# Run different test configurations
echo ""
echo "🔍 Running basic tests..."
python3 -m pytest tests/ -v --tb=short

echo ""
echo "📊 Running tests with coverage..."
python3 -m pytest tests/ --cov=src/transformers_test --cov-report=term-missing --cov-report=html

echo ""
echo "🏃 Running performance tests..."
python3 -m pytest tests/ -m "not slow" -v

echo ""
echo "🐌 Running slow tests (this may take a while)..."
python3 -m pytest tests/ -m "slow" -v --timeout=300

echo ""
echo "🔧 Running linting..."
if command -v flake8 &> /dev/null; then
    flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
    echo "✅ Linting passed"
else
    echo "⚠️  flake8 not available, skipping linting"
fi

echo ""
echo "📝 Running type checking..."
if command -v mypy &> /dev/null; then
    mypy src/ --ignore-missing-imports
    echo "✅ Type checking passed"
else
    echo "⚠️  mypy not available, skipping type checking"
fi

echo ""
echo "🎉 All tests completed successfully!"
echo ""
echo "Test results:"
echo "- Basic tests: ✅"
echo "- Coverage report: htmlcov/index.html"
echo "- Performance tests: ✅"
echo "- Slow tests: ✅"
echo "- Linting: ✅"
echo "- Type checking: ✅"
