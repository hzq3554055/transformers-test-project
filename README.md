# Transformers Testing Framework

A comprehensive testing framework for Hugging Face Transformers models, designed to facilitate model evaluation, comparison, and performance analysis.

## Features

- **Model Testing**: Easy-to-use interface for testing transformer models
- **Performance Benchmarking**: Comprehensive performance analysis and comparison
- **Data Processing**: Utilities for dataset preparation and tokenization
- **Model Training**: Built-in training capabilities with best practices
- **Interactive Notebooks**: Jupyter notebooks for exploration and experimentation
- **CLI Interface**: Command-line tools for quick testing and training
- **Extensive Testing**: Full test suite with pytest

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd transformers_test_project
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

## Quick Start

### Using Python API

```python
from transformers_test import ModelTester, DataProcessor, setup_logging

# Setup logging
logger = setup_logging(level='INFO')

# Test a model
tester = ModelTester(
    model_name='bert-base-uncased',
    task_type='classification'
)

# Run inference
result = tester.test_inference("Hello world!")
print(f"Predictions: {result['predictions']}")

# Benchmark performance
benchmark = tester.benchmark_performance(
    text="This is a test sentence.",
    num_runs=10
)
print(f"Mean inference time: {benchmark['mean_time']:.4f}s")
```

### Using CLI

```bash
# Test model inference
transformers-test test --model bert-base-uncased --text "Hello world"

# Train a model
transformers-test train --model bert-base-uncased --data data/train.json

# Benchmark performance
transformers-test benchmark --model bert-base-uncased --text "Hello world" --runs 10
```

### Using Jupyter Notebooks

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/01_quick_start.ipynb` for a comprehensive tutorial

## Project Structure

```
transformers_test_project/
├── src/
│   └── transformers_test/
│       ├── __init__.py
│       ├── models.py          # Model testing utilities
│       ├── data.py            # Data processing utilities
│       ├── training.py        # Training utilities
│       ├── utils.py           # Utility functions
│       └── cli.py             # Command-line interface
├── tests/                     # Test suite
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_training.py
│   └── test_utils.py
├── notebooks/                 # Jupyter notebooks
│   ├── 01_quick_start.ipynb
│   └── 02_model_comparison.ipynb
├── data/                      # Data directory
├── models/                    # Model checkpoints
├── configs/                   # Configuration files
├── scripts/                   # Utility scripts
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
├── pyproject.toml            # Project configuration
├── setup.py                  # Setup script
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Core Components

### ModelTester

The `ModelTester` class provides comprehensive model testing capabilities:

```python
from transformers_test import ModelTester

tester = ModelTester(
    model_name='bert-base-uncased',
    task_type='classification'
)

# Test inference
result = tester.test_inference("Your text here")

# Batch inference
results = tester.test_batch_inference(["Text 1", "Text 2"])

# Performance benchmarking
benchmark = tester.benchmark_performance("Test text", num_runs=10)

# Get model information
info = tester.get_model_info()
```

### DataProcessor

The `DataProcessor` class handles data preparation and tokenization:

```python
from transformers_test import DataProcessor

processor = DataProcessor(
    tokenizer_name='bert-base-uncased',
    max_length=512
)

# Create sample data
dataset = processor.create_sample_data(num_samples=100)

# Tokenize dataset
tokenized_dataset = processor.tokenize_dataset(dataset)

# Create train-test split
train_dataset, test_dataset = processor.create_train_test_split(dataset)

# Create DataLoader
dataloader = processor.create_data_loader(tokenized_dataset, batch_size=16)
```

### ModelTrainer

The `ModelTrainer` class provides training capabilities:

```python
from transformers_test import ModelTrainer

trainer = ModelTrainer(
    model_name='bert-base-uncased',
    task_type='classification',
    num_labels=2
)

# Train model
trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    output_dir='./results',
    num_train_epochs=3
)

# Evaluate model
eval_results = trainer.evaluate(eval_dataset)

# Make predictions
predictions = trainer.predict(test_dataset)
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```env
# Hugging Face Hub
HF_TOKEN=your_huggingface_token_here

# Model configuration
MODEL_NAME=bert-base-uncased
MAX_LENGTH=512
BATCH_SIZE=16
LEARNING_RATE=2e-5

# Data paths
DATA_DIR=./data
MODEL_DIR=./models
LOG_DIR=./logs
```

### Configuration Files

Generate a configuration file:

```bash
transformers-test config --output config.yaml
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/transformers_test

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks

Setup pre-commit hooks:

```bash
pre-commit install
```

### Running Quality Checks

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Examples

### Model Comparison

Compare different transformer models:

```python
from transformers_test import ModelTester
import pandas as pd

models = ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base']
results = []

for model_name in models:
    tester = ModelTester(model_name=model_name)
    
    # Benchmark performance
    benchmark = tester.benchmark_performance("Test sentence", num_runs=5)
    
    # Get model info
    info = tester.get_model_info()
    
    results.append({
        'model': model_name,
        'mean_time': benchmark['mean_time'],
        'parameters': info['num_parameters'],
        'size_mb': info['model_size_mb']
    })

# Create comparison DataFrame
df = pd.DataFrame(results)
print(df)
```

### Custom Training

Train a model on your own data:

```python
from transformers_test import ModelTrainer, DataProcessor

# Prepare data
processor = DataProcessor('bert-base-uncased')
dataset = processor.load_dataset_from_file('your_data.json')
tokenized_dataset = processor.tokenize_dataset(dataset)

# Train model
trainer = ModelTrainer('bert-base-uncased', num_labels=2)
trainer.train(
    train_dataset=tokenized_dataset,
    output_dir='./my_model',
    num_train_epochs=3
)
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Commit your changes: `git commit -am 'Add feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the example notebooks in `notebooks/`

## Changelog

### Version 0.1.0

- Initial release
- Core testing framework
- Model comparison utilities
- Training capabilities
- CLI interface
- Jupyter notebook examples
- Comprehensive test suite
