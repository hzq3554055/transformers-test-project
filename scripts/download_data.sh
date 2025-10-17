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
