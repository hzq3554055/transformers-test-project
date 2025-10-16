"""
Data processing utilities for transformers testing.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processing utilities for transformers testing.
    """
    
    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 512,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the DataProcessor.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            cache_dir: Directory to cache tokenizers
        """
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        self.tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                cache_dir=self.cache_dir
            )
            logger.info(f"Tokenizer loaded: {self.tokenizer_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def load_dataset_from_hf(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        streaming: bool = False
    ) -> Dataset:
        """
        Load dataset from Hugging Face Hub.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split to load
            subset: Subset of the dataset
            streaming: Whether to use streaming
            
        Returns:
            Loaded dataset
        """
        try:
            if subset:
                dataset = load_dataset(
                    dataset_name,
                    subset,
                    split=split,
                    streaming=streaming,
                    cache_dir=self.cache_dir
                )
            else:
                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=streaming,
                    cache_dir=self.cache_dir
                )
            
            logger.info(f"Dataset loaded: {dataset_name} ({split})")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def load_dataset_from_file(
        self,
        filepath: str,
        file_type: str = "auto"
    ) -> Dataset:
        """
        Load dataset from local file.
        
        Args:
            filepath: Path to the data file
            file_type: Type of file (json, csv, txt, auto)
            
        Returns:
            Loaded dataset
        """
        try:
            filepath = Path(filepath)
            
            if file_type == "auto":
                file_type = filepath.suffix.lower().lstrip('.')
            
            if file_type == "json":
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                dataset = Dataset.from_list(data)
            elif file_type == "csv":
                df = pd.read_csv(filepath)
                dataset = Dataset.from_pandas(df)
            elif file_type == "txt":
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                data = [{"text": line.strip()} for line in lines if line.strip()]
                dataset = Dataset.from_list(data)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"Dataset loaded from file: {filepath}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset from file: {e}")
            raise
    
    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        label_column: Optional[str] = None,
        padding: str = "max_length",
        truncation: bool = True
    ) -> Dataset:
        """
        Tokenize a dataset.
        
        Args:
            dataset: Input dataset
            text_column: Name of the text column
            label_column: Name of the label column
            padding: Padding strategy
            truncation: Whether to truncate sequences
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding=padding,
                truncation=truncation,
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        try:
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            logger.info("Dataset tokenized successfully")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Failed to tokenize dataset: {e}")
            raise
    
    def create_train_test_split(
        self,
        dataset: Dataset,
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[Dataset, Dataset]:
        """
        Create train-test split.
        
        Args:
            dataset: Input dataset
            test_size: Proportion of data for testing
            seed: Random seed
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        try:
            split_dataset = dataset.train_test_split(
                test_size=test_size,
                seed=seed
            )
            
            train_dataset = split_dataset["train"]
            test_dataset = split_dataset["test"]
            
            logger.info(f"Dataset split: {len(train_dataset)} train, {len(test_dataset)} test")
            return train_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Failed to create train-test split: {e}")
            raise
    
    def create_data_loader(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """
        Create a PyTorch DataLoader.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader
        """
        from torch.utils.data import DataLoader
        
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self._collate_fn
            )
            
            logger.info(f"DataLoader created: batch_size={batch_size}")
            return dataloader
            
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}")
            raise
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        import torch
        
        # Stack input_ids and attention_mask
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Add labels if present
        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
            result["labels"] = labels
        
        return result
    
    def save_dataset(
        self,
        dataset: Dataset,
        filepath: str,
        format: str = "json"
    ):
        """
        Save dataset to file.
        
        Args:
            dataset: Dataset to save
            filepath: Output file path
            format: Output format (json, csv, parquet)
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                dataset.to_json(filepath)
            elif format == "csv":
                dataset.to_csv(filepath)
            elif format == "parquet":
                dataset.to_parquet(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Dataset saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset information
        """
        return {
            "num_examples": len(dataset),
            "features": list(dataset.features.keys()),
            "column_names": dataset.column_names,
            "info": dataset.info if hasattr(dataset, 'info') else None
        }
    
    def create_sample_data(
        self,
        num_samples: int = 100,
        text_length: int = 100
    ) -> Dataset:
        """
        Create sample data for testing.
        
        Args:
            num_samples: Number of samples to create
            text_length: Length of each text sample
            
        Returns:
            Sample dataset
        """
        import random
        import string
        
        def generate_text(length):
            return ''.join(random.choices(string.ascii_letters + ' ', k=length))
        
        data = []
        for i in range(num_samples):
            data.append({
                "text": generate_text(text_length),
                "label": random.randint(0, 1),
                "id": i
            })
        
        dataset = Dataset.from_list(data)
        logger.info(f"Sample dataset created: {num_samples} samples")
        return dataset
