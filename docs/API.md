# API Reference

This document provides detailed API reference for the transformers testing framework.

## Core Classes

### ModelTester

The main class for testing transformer models.

#### `__init__(model_name, task_type='classification', device=None, cache_dir=None)`

Initialize the ModelTester.

**Parameters:**
- `model_name` (str): Name or path of the model to test
- `task_type` (str): Type of task (classification, token_classification, qa, etc.)
- `device` (str, optional): Device to run on (auto-detected if None)
- `cache_dir` (str, optional): Directory to cache models

**Example:**
```python
tester = ModelTester(
    model_name='bert-base-uncased',
    task_type='classification'
)
```

#### `test_inference(text, max_length=512, return_tensors='pt')`

Test model inference on given text.

**Parameters:**
- `text` (str or List[str]): Input text or list of texts
- `max_length` (int): Maximum sequence length
- `return_tensors` (str): Format of returned tensors

**Returns:**
- `Dict[str, Any]`: Dictionary containing model outputs and metadata

**Example:**
```python
result = tester.test_inference("Hello world!")
print(result['predictions'])
```

#### `test_batch_inference(texts, batch_size=8, max_length=512)`

Test batch inference on multiple texts.

**Parameters:**
- `texts` (List[str]): List of input texts
- `batch_size` (int): Size of each batch
- `max_length` (int): Maximum sequence length

**Returns:**
- `List[Dict[str, Any]]`: List of inference results

#### `benchmark_performance(text, num_runs=10, warmup_runs=3)`

Benchmark model performance.

**Parameters:**
- `text` (str): Input text for benchmarking
- `num_runs` (int): Number of benchmark runs
- `warmup_runs` (int): Number of warmup runs

**Returns:**
- `Dict[str, float]`: Performance metrics

#### `get_model_info()`

Get comprehensive model information.

**Returns:**
- `Dict[str, Any]`: Model information including parameters, size, etc.

#### `save_test_results(results, filepath)`

Save test results to file.

**Parameters:**
- `results` (Dict[str, Any]): Test results to save
- `filepath` (str): Path to save results

### DataProcessor

Class for data processing and preparation.

#### `__init__(tokenizer_name, max_length=512, cache_dir=None)`

Initialize the DataProcessor.

**Parameters:**
- `tokenizer_name` (str): Name of the tokenizer to use
- `max_length` (int): Maximum sequence length
- `cache_dir` (str, optional): Directory to cache tokenizers

#### `load_dataset_from_hf(dataset_name, split='train', subset=None, streaming=False)`

Load dataset from Hugging Face Hub.

**Parameters:**
- `dataset_name` (str): Name of the dataset
- `split` (str): Dataset split to load
- `subset` (str, optional): Subset of the dataset
- `streaming` (bool): Whether to use streaming

**Returns:**
- `Dataset`: Loaded dataset

#### `load_dataset_from_file(filepath, file_type='auto')`

Load dataset from local file.

**Parameters:**
- `filepath` (str): Path to the data file
- `file_type` (str): Type of file (json, csv, txt, auto)

**Returns:**
- `Dataset`: Loaded dataset

#### `tokenize_dataset(dataset, text_column='text', label_column=None, padding='max_length', truncation=True)`

Tokenize a dataset.

**Parameters:**
- `dataset` (Dataset): Input dataset
- `text_column` (str): Name of the text column
- `label_column` (str, optional): Name of the label column
- `padding` (str): Padding strategy
- `truncation` (bool): Whether to truncate sequences

**Returns:**
- `Dataset`: Tokenized dataset

#### `create_train_test_split(dataset, test_size=0.2, seed=42)`

Create train-test split.

**Parameters:**
- `dataset` (Dataset): Input dataset
- `test_size` (float): Proportion of data for testing
- `seed` (int): Random seed

**Returns:**
- `Tuple[Dataset, Dataset]`: Tuple of (train_dataset, test_dataset)

#### `create_data_loader(dataset, batch_size=16, shuffle=True, num_workers=0)`

Create a PyTorch DataLoader.

**Parameters:**
- `dataset` (Dataset): Input dataset
- `batch_size` (int): Batch size
- `shuffle` (bool): Whether to shuffle data
- `num_workers` (int): Number of worker processes

**Returns:**
- `DataLoader`: PyTorch DataLoader

### ModelTrainer

Class for training transformer models.

#### `__init__(model_name, task_type='classification', num_labels=2, device=None, cache_dir=None)`

Initialize the ModelTrainer.

**Parameters:**
- `model_name` (str): Name of the model to train
- `task_type` (str): Type of task (classification, token_classification, qa)
- `num_labels` (int): Number of labels for classification
- `device` (str, optional): Device to train on
- `cache_dir` (str, optional): Directory to cache models

#### `train(train_dataset, eval_dataset=None, output_dir='./results', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16, learning_rate=2e-5, weight_decay=0.01, warmup_steps=500, logging_steps=100, eval_steps=500, save_steps=500, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, early_stopping_patience=3, **kwargs)`

Train the model.

**Parameters:**
- `train_dataset`: Training dataset
- `eval_dataset`: Evaluation dataset (optional)
- `output_dir` (str): Output directory for checkpoints
- `num_train_epochs` (int): Number of training epochs
- `per_device_train_batch_size` (int): Training batch size
- `per_device_eval_batch_size` (int): Evaluation batch size
- `learning_rate` (float): Learning rate
- `weight_decay` (float): Weight decay
- `warmup_steps` (int): Number of warmup steps
- `logging_steps` (int): Logging frequency
- `eval_steps` (int): Evaluation frequency
- `save_steps` (int): Save frequency
- `save_total_limit` (int): Maximum number of checkpoints to keep
- `load_best_model_at_end` (bool): Whether to load best model at end
- `metric_for_best_model` (str): Metric for model selection
- `greater_is_better` (bool): Whether higher metric is better
- `early_stopping_patience` (int): Early stopping patience
- `**kwargs`: Additional training arguments

**Returns:**
- `Trainer`: Trained trainer

#### `evaluate(eval_dataset, metric_key_prefix='eval')`

Evaluate the model.

**Parameters:**
- `eval_dataset`: Evaluation dataset
- `metric_key_prefix` (str): Prefix for metric keys

**Returns:**
- `Dict[str, float]`: Evaluation metrics

#### `predict(test_dataset, return_predictions=True)`

Make predictions on test dataset.

**Parameters:**
- `test_dataset`: Test dataset
- `return_predictions` (bool): Whether to return predictions

**Returns:**
- `Union[Tuple, np.ndarray]`: Predictions or tuple of (predictions, labels)

#### `save_model(output_dir)`

Save the trained model.

**Parameters:**
- `output_dir` (str): Directory to save the model

#### `load_model(model_path)`

Load a trained model.

**Parameters:**
- `model_path` (str): Path to the saved model

## Utility Functions

### `setup_logging(level='INFO', log_file=None, format_string=None)`

Setup logging configuration.

**Parameters:**
- `level` (str): Logging level
- `log_file` (str, optional): Optional log file path
- `format_string` (str, optional): Custom format string

**Returns:**
- `logging.Logger`: Configured logger

### `get_device(device=None)`

Get the best available device.

**Parameters:**
- `device` (str, optional): Preferred device

**Returns:**
- `str`: Device string

### `set_seed(seed=42)`

Set random seed for reproducibility.

**Parameters:**
- `seed` (int): Random seed

### `load_config(config_path)`

Load configuration from file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

### `save_config(config, config_path)`

Save configuration to file.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary
- `config_path` (str): Path to save configuration

### `create_experiment_dir(base_dir, experiment_name=None)`

Create experiment directory with timestamp.

**Parameters:**
- `base_dir` (str): Base directory for experiments
- `experiment_name` (str, optional): Optional experiment name

**Returns:**
- `str`: Path to created directory

### `get_model_size(model)`

Get model size information.

**Parameters:**
- `model` (torch.nn.Module): PyTorch model

**Returns:**
- `Dict[str, float]`: Dictionary with size information

### `count_tokens(text, tokenizer)`

Count tokens in text.

**Parameters:**
- `text` (str): Input text
- `tokenizer`: Tokenizer to use

**Returns:**
- `int`: Number of tokens

### `batch_texts(texts, batch_size)`

Split texts into batches.

**Parameters:**
- `texts` (List[str]): List of texts
- `batch_size` (int): Batch size

**Returns:**
- `List[List[str]]`: List of text batches

## CLI Commands

### `transformers-test test`

Test model inference.

**Options:**
- `--model`: Model name or path (required)
- `--text`: Input text (required)
- `--task`: Task type (default: classification)
- `--device`: Device to use
- `--max-length`: Max sequence length (default: 512)

**Example:**
```bash
transformers-test test --model bert-base-uncased --text "Hello world"
```

### `transformers-test train`

Train a model.

**Options:**
- `--model`: Model name or path (required)
- `--data`: Training data path (required)
- `--output`: Output directory (default: ./results)
- `--epochs`: Number of epochs (default: 3)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--device`: Device to use

**Example:**
```bash
transformers-test train --model bert-base-uncased --data data/train.json
```

### `transformers-test benchmark`

Benchmark model performance.

**Options:**
- `--model`: Model name or path (required)
- `--text`: Input text (required)
- `--runs`: Number of benchmark runs (default: 10)
- `--warmup`: Number of warmup runs (default: 3)
- `--device`: Device to use

**Example:**
```bash
transformers-test benchmark --model bert-base-uncased --text "Hello world" --runs 10
```

### `transformers-test config`

Generate configuration file.

**Options:**
- `--output`: Output config file (default: config.yaml)

**Example:**
```bash
transformers-test config --output my_config.yaml
```

## Error Handling

The framework includes comprehensive error handling:

- **Model Loading Errors**: Clear error messages for invalid model names or paths
- **Data Processing Errors**: Validation for data format and structure
- **Training Errors**: Graceful handling of training failures
- **Device Errors**: Automatic fallback to CPU if GPU is unavailable

## Performance Considerations

- **Memory Management**: Automatic cleanup of unused tensors
- **Batch Processing**: Efficient batch inference for multiple texts
- **Caching**: Model and tokenizer caching to avoid redundant downloads
- **Device Optimization**: Automatic device selection and optimization

## Best Practices

1. **Use appropriate batch sizes** for your hardware
2. **Set random seeds** for reproducible results
3. **Monitor memory usage** during training
4. **Use validation datasets** for model evaluation
5. **Save checkpoints** regularly during training
6. **Use appropriate sequence lengths** for your task
