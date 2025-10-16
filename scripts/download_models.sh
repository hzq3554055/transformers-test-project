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
