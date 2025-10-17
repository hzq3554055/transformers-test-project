# Transformers Testing Framework - Project Info

## Project Structure
```
transformers_test_project/
├── src/transformers_test/     # Main source code
├── tests/                     # Test suite
├── notebooks/                 # Jupyter notebooks
├── data/                     # Data directory
├── models/                   # Model checkpoints
├── configs/                  # Configuration files
├── scripts/                  # Utility scripts
├── docs/                     # Documentation
├── results/                  # Results and outputs
└── logs/                     # Log files
```

## Quick Start Commands

### Setup
```bash
# Install dependencies
./scripts/install_dependencies.sh

# Setup environment
./scripts/setup_environment.sh

# Setup development environment
./scripts/dev_setup.sh
```

### Usage
```bash
# Run tests
./scripts/run_tests.sh

# Benchmark models
./scripts/benchmark_models.sh

# Download sample data
./scripts/download_data.sh

# Download models
./scripts/download_models.sh
```

### Development
```bash
# Start Jupyter
jupyter notebook

# Run CLI
transformers-test test --model bert-base-uncased --text "Hello world"

# Clean up
./scripts/cleanup.sh
```

## Configuration

- Edit `.env` for environment variables
- Edit `configs/` for model and training configuration
- Edit `requirements.txt` for dependencies

## Documentation

- `README.md` - Main documentation
- `docs/API.md` - API reference
- `notebooks/` - Interactive examples
