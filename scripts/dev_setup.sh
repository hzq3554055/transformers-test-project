#!/bin/bash
# Development environment setup

echo "🛠️  Setting up development environment..."

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit
pip install pre-commit
pre-commit install

# Install Jupyter extensions
pip install jupyterlab
pip install ipywidgets

# Install additional development tools
pip install black flake8 mypy pytest-cov

echo "✅ Development environment ready"
