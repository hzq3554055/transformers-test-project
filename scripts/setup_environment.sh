#!/bin/bash

# Setup Environment Script
# This script sets up the complete development environment for the transformers testing framework

set -e  # Exit on any error

echo "🔧 Setting up Transformers Testing Framework Environment"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed}
mkdir -p models
mkdir -p configs
mkdir -p scripts
mkdir -p docs
mkdir -p results/{benchmarks,experiments,logs}
mkdir -p logs

echo "✅ Project directories created"

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Environment file created from template"
        echo "⚠️  Please edit .env file with your configuration"
    else
        echo "⚠️  .env.example not found, creating basic .env file"
        cat > .env << EOF
# Environment variables for transformers testing project

# Hugging Face Hub
HF_TOKEN=your_huggingface_token_here
HF_HOME=./cache

# Model configuration
MODEL_NAME=bert-base-uncased
MAX_LENGTH=512
BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=3

# Data paths
DATA_DIR=./data
MODEL_DIR=./models
LOG_DIR=./logs

# Training configuration
DEVICE=auto
NUM_WORKERS=4
SEED=42

# Logging
LOG_LEVEL=INFO
WANDB_PROJECT=transformers-test
WANDB_ENTITY=your_username
EOF
        echo "✅ Basic .env file created"
    fi
else
    echo "✅ .env file already exists"
fi

# Create sample configuration files
echo "⚙️  Creating sample configuration files..."

# Create sample training config
cat > configs/training_config.yaml << EOF
model:
  name: bert-base-uncased
  task_type: classification
  max_length: 512
  device: auto

training:
  num_epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 500
  logging_steps: 100
  eval_steps: 500
  save_steps: 500

data:
  train_path: data/train.json
  eval_path: data/eval.json
  test_path: data/test.json

logging:
  level: INFO
  log_file: logs/experiment.log
EOF

# Create sample data config
cat > configs/data_config.yaml << EOF
dataset:
  name: imdb
  split: train
  streaming: false

preprocessing:
  max_length: 512
  padding: max_length
  truncation: true

splits:
  train_size: 0.8
  eval_size: 0.1
  test_size: 0.1
  seed: 42
EOF

echo "✅ Configuration files created"

# Create sample data files
echo "📊 Creating sample data files..."

# Create sample training data
cat > data/train.json << EOF
[
  {"text": "This is a positive review.", "label": 1},
  {"text": "This is a negative review.", "label": 0},
  {"text": "I love this movie!", "label": 1},
  {"text": "This movie is terrible.", "label": 0},
  {"text": "Great acting and story.", "label": 1},
  {"text": "Waste of time.", "label": 0},
  {"text": "Amazing cinematography.", "label": 1},
  {"text": "Boring and slow.", "label": 0},
  {"text": "Highly recommended!", "label": 1},
  {"text": "Not worth watching.", "label": 0}
]
EOF

# Create sample evaluation data
cat > data/eval.json << EOF
[
  {"text": "Excellent movie with great acting.", "label": 1},
  {"text": "Poor storyline and bad acting.", "label": 0},
  {"text": "One of the best films I've seen.", "label": 1},
  {"text": "Complete waste of money.", "label": 0},
  {"text": "Outstanding performance by the cast.", "label": 1}
]
EOF

echo "✅ Sample data files created"

# Create utility scripts
echo "🛠️  Creating utility scripts..."

# Create data download script
cat > scripts/download_data.sh << 'EOF'
#!/bin/bash
# Download sample datasets for testing

echo "📥 Downloading sample datasets..."

# Download IMDB dataset
python3 -c "
from datasets import load_dataset
import json

# Load IMDB dataset
dataset = load_dataset('imdb')

# Save as JSON files
dataset['train'].to_json('data/imdb_train.json')
dataset['test'].to_json('data/imdb_test.json')

print('✅ IMDB dataset downloaded and saved')
"

# Download other sample datasets
python3 -c "
from datasets import load_dataset
import json

# Load other datasets
datasets = [
    ('glue', 'sst2'),
    ('ag_news', None),
    ('yelp_polarity', None)
]

for dataset_name, subset in datasets:
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)
        
        # Save train split
        if 'train' in dataset:
            dataset['train'].to_json(f'data/{dataset_name.replace(\"-\", \"_\")}_train.json')
            print(f'✅ {dataset_name} dataset downloaded')
    except Exception as e:
        print(f'⚠️  Could not download {dataset_name}: {e}')
"

echo "✅ Data download completed"
EOF

chmod +x scripts/download_data.sh

# Create model download script
cat > scripts/download_models.sh << 'EOF'
#!/bin/bash
# Download and cache popular transformer models

echo "🤖 Downloading and caching transformer models..."

MODELS=(
    "bert-base-uncased"
    "distilbert-base-uncased"
    "roberta-base"
    "albert-base-v2"
    "xlnet-base-cased"
)

for model in "${MODELS[@]}"; do
    echo "  Downloading $model..."
    python3 -c "
from transformers import AutoTokenizer, AutoModel
import sys

model_name = sys.argv[1]
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print(f'    ✅ {model_name} downloaded successfully')
except Exception as e:
    print(f'    ❌ Error downloading {model_name}: {e}')
" "$model"
done

echo "✅ Model download completed"
EOF

chmod +x scripts/download_models.sh

# Create cleanup script
cat > scripts/cleanup.sh << 'EOF'
#!/bin/bash
# Clean up temporary files and caches

echo "🧹 Cleaning up temporary files..."

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Remove test artifacts
rm -rf .pytest_cache/
rm -rf htmlcov/
rm -rf .coverage
rm -rf .mypy_cache/

# Remove temporary files
rm -rf /tmp/transformers_test_*

# Clean up logs (keep last 10)
find logs/ -name "*.log" -type f | head -n -10 | xargs rm -f 2>/dev/null || true

echo "✅ Cleanup completed"
EOF

chmod +x scripts/cleanup.sh

echo "✅ Utility scripts created"

# Set up Git hooks
echo "🔗 Setting up Git hooks..."

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for transformers testing framework

echo "🔍 Running pre-commit checks..."

# Run linting
if command -v flake8 &> /dev/null; then
    flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
    if [ $? -ne 0 ]; then
        echo "❌ Linting failed"
        exit 1
    fi
fi

# Run type checking
if command -v mypy &> /dev/null; then
    mypy src/ --ignore-missing-imports
    if [ $? -ne 0 ]; then
        echo "❌ Type checking failed"
        exit 1
    fi
fi

# Run tests
python3 -m pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo "❌ Tests failed"
    exit 1
fi

echo "✅ Pre-commit checks passed"
EOF

chmod +x .git/hooks/pre-commit

echo "✅ Git hooks set up"

# Create development setup script
cat > scripts/dev_setup.sh << 'EOF'
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
EOF

chmod +x scripts/dev_setup.sh

echo "✅ Development setup script created"

# Create project info file
cat > PROJECT_INFO.md << EOF
# Transformers Testing Framework - Project Info

## Project Structure
\`\`\`
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
\`\`\`

## Quick Start Commands

### Setup
\`\`\`bash
# Install dependencies
./scripts/install_dependencies.sh

# Setup environment
./scripts/setup_environment.sh

# Setup development environment
./scripts/dev_setup.sh
\`\`\`

### Usage
\`\`\`bash
# Run tests
./scripts/run_tests.sh

# Benchmark models
./scripts/benchmark_models.sh

# Download sample data
./scripts/download_data.sh

# Download models
./scripts/download_models.sh
\`\`\`

### Development
\`\`\`bash
# Start Jupyter
jupyter notebook

# Run CLI
transformers-test test --model bert-base-uncased --text "Hello world"

# Clean up
./scripts/cleanup.sh
\`\`\`

## Configuration

- Edit \`.env\` for environment variables
- Edit \`configs/\` for model and training configuration
- Edit \`requirements.txt\` for dependencies

## Documentation

- \`README.md\` - Main documentation
- \`docs/API.md\` - API reference
- \`notebooks/\` - Interactive examples
EOF

echo "✅ Project info file created"

echo ""
echo "🎉 Environment setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: ./scripts/install_dependencies.sh"
echo "3. Run: ./scripts/dev_setup.sh (for development)"
echo "4. Start Jupyter: jupyter notebook"
echo "5. Or test CLI: transformers-test test --model bert-base-uncased --text 'Hello world'"
echo ""
echo "📚 Documentation:"
echo "- README.md - Main documentation"
echo "- docs/API.md - API reference"
echo "- notebooks/ - Interactive examples"
echo "- PROJECT_INFO.md - Project information"
echo ""
echo "🛠️  Available scripts:"
echo "- install_dependencies.sh - Install all dependencies"
echo "- run_tests.sh - Run test suite"
echo "- benchmark_models.sh - Benchmark model performance"
echo "- download_data.sh - Download sample datasets"
echo "- download_models.sh - Download and cache models"
echo "- cleanup.sh - Clean up temporary files"
echo "- dev_setup.sh - Setup development environment"
