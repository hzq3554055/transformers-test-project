"""
Command-line interface for transformers testing.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .models import ModelTester
from .data import DataProcessor
from .training import ModelTrainer
from .utils import setup_logging, set_seed, load_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transformers Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a model
  transformers-test test --model bert-base-uncased --text "Hello world"
  
  # Train a model
  transformers-test train --model bert-base-uncased --data data/train.json
  
  # Benchmark model performance
  transformers-test benchmark --model bert-base-uncased --text "Hello world" --runs 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test model inference')
    test_parser.add_argument('--model', required=True, help='Model name or path')
    test_parser.add_argument('--text', required=True, help='Input text')
    test_parser.add_argument('--task', default='classification', help='Task type')
    test_parser.add_argument('--device', help='Device to use')
    test_parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', required=True, help='Model name or path')
    train_parser.add_argument('--data', required=True, help='Training data path')
    train_parser.add_argument('--output', default='./results', help='Output directory')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    train_parser.add_argument('--device', help='Device to use')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark model performance')
    benchmark_parser.add_argument('--model', required=True, help='Model name or path')
    benchmark_parser.add_argument('--text', required=True, help='Input text')
    benchmark_parser.add_argument('--runs', type=int, default=10, help='Number of benchmark runs')
    benchmark_parser.add_argument('--warmup', type=int, default=3, help='Number of warmup runs')
    benchmark_parser.add_argument('--device', help='Device to use')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Generate configuration file')
    config_parser.add_argument('--output', default='config.yaml', help='Output config file')
    
    # Global arguments
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(
        level=args.log_level,
        log_file=args.log_file
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    try:
        if args.command == 'test':
            run_test(args, config)
        elif args.command == 'train':
            run_train(args, config)
        elif args.command == 'benchmark':
            run_benchmark(args, config)
        elif args.command == 'config':
            generate_config(args)
        else:
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


def run_test(args, config: dict):
    """Run model testing."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Testing model: {args.model}")
    logger.info(f"Input text: {args.text}")
    
    # Initialize model tester
    tester = ModelTester(
        model_name=args.model,
        task_type=args.task,
        device=args.device
    )
    
    # Run inference
    result = tester.test_inference(
        text=args.text,
        max_length=args.max_length
    )
    
    # Print results
    print(f"Model: {result['model_name']}")
    print(f"Device: {result['device']}")
    print(f"Input length: {result['input_length']}")
    
    if args.task == 'classification':
        predictions = result['predictions']
        predicted_class = result['predicted_class']
        print(f"Predictions: {predictions}")
        print(f"Predicted class: {predicted_class}")
    
    logger.info("Test completed successfully")


def run_train(args, config: dict):
    """Run model training."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Training model: {args.model}")
    logger.info(f"Data: {args.data}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_name=args.model,
        device=args.device
    )
    
    # Load data
    data_processor = DataProcessor(
        tokenizer_name=args.model
    )
    
    # For now, create sample data
    # In a real implementation, you would load from args.data
    dataset = data_processor.create_sample_data(num_samples=100)
    
    # Split data
    train_dataset, eval_dataset = data_processor.create_train_test_split(dataset)
    
    # Tokenize data
    train_dataset = data_processor.tokenize_dataset(train_dataset)
    eval_dataset = data_processor.tokenize_dataset(eval_dataset)
    
    # Train model
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    logger.info("Training completed successfully")


def run_benchmark(args, config: dict):
    """Run model benchmarking."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Benchmarking model: {args.model}")
    logger.info(f"Input text: {args.text}")
    
    # Initialize model tester
    tester = ModelTester(
        model_name=args.model,
        device=args.device
    )
    
    # Run benchmark
    results = tester.benchmark_performance(
        text=args.text,
        num_runs=args.runs,
        warmup_runs=args.warmup
    )
    
    # Print results
    print(f"Benchmark Results:")
    print(f"  Mean time: {results['mean_time']:.4f}s")
    print(f"  Min time: {results['min_time']:.4f}s")
    print(f"  Max time: {results['max_time']:.4f}s")
    print(f"  Std time: {results['std_time']:.4f}s")
    print(f"  Runs: {results['runs']}")
    
    logger.info("Benchmark completed successfully")


def generate_config(args):
    """Generate configuration file."""
    import yaml
    
    config = {
        'model': {
            'name': 'bert-base-uncased',
            'task_type': 'classification',
            'max_length': 512,
            'device': 'auto'
        },
        'training': {
            'num_epochs': 3,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_steps': 500
        },
        'data': {
            'train_path': 'data/train.json',
            'eval_path': 'data/eval.json',
            'test_path': 'data/test.json'
        },
        'logging': {
            'level': 'INFO',
            'log_file': 'logs/experiment.log'
        }
    }
    
    with open(args.output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration file generated: {args.output}")


if __name__ == '__main__':
    main()
