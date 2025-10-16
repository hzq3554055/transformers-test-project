#!/bin/bash

# Install Dependencies Script
# This script installs all required dependencies for the transformers testing framework

set -e  # Exit on any error

echo "🚀 Installing Transformers Testing Framework Dependencies"
echo "=========================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION is installed, but Python $REQUIRED_VERSION or higher is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 detected"

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch (CPU version by default, GPU if CUDA is available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU detected, installing PyTorch with CUDA support"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "  No GPU detected, installing PyTorch CPU version"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
echo "📚 Installing core dependencies..."
pip3 install -r requirements.txt

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip3 install -e ".[dev]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  Pre-commit not available, skipping hooks setup"
fi

# Verify installation
echo "🧪 Verifying installation..."
python3 -c "
import torch
import transformers
import datasets
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ Transformers version: {transformers.__version__}')
print(f'✅ Datasets version: {datasets.__version__}')
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('ℹ️  CUDA not available, using CPU')
"

# Run tests
echo "🧪 Running tests..."
if python3 -m pytest tests/ -v; then
    echo "✅ All tests passed"
else
    echo "⚠️  Some tests failed, but installation is complete"
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate your virtual environment: source venv/bin/activate"
echo "2. Run the quick start notebook: jupyter notebook notebooks/01_quick_start.ipynb"
echo "3. Or test the CLI: transformers-test test --model bert-base-uncased --text 'Hello world'"
echo ""
echo "For more information, see README.md"
