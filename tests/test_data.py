"""
Tests for data processing utilities.
"""

import pytest
import tempfile
import json
from pathlib import Path
from transformers_test.data import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor(
            tokenizer_name="bert-base-uncased",
            max_length=512
        )
        
        assert processor.tokenizer_name == "bert-base-uncased"
        assert processor.max_length == 512
        assert processor.tokenizer is not None
    
    def test_create_sample_data(self):
        """Test creating sample data."""
        processor = DataProcessor(
            tokenizer_name="bert-base-uncased",
            max_length=512
        )
        
        dataset = processor.create_sample_data(num_samples=10, text_length=50)
        
        assert len(dataset) == 10
        assert "text" in dataset.features
        assert "label" in dataset.features
        assert "id" in dataset.features
    
    def test_tokenize_dataset(self):
        """Test tokenizing dataset."""
        processor = DataProcessor(
            tokenizer_name="bert-base-uncased",
            max_length=512
        )
        
        dataset = processor.create_sample_data(num_samples=5, text_length=50)
        tokenized_dataset = processor.tokenize_dataset(dataset)
        
        assert "input_ids" in tokenized_dataset.features
        assert "attention_mask" in tokenized_dataset.features
        assert len(tokenized_dataset) == 5
    
    def test_create_train_test_split(self):
        """Test creating train-test split."""
        processor = DataProcessor(
            tokenizer_name="bert-base-uncased",
            max_length=512
        )
        
        dataset = processor.create_sample_data(num_samples=100, text_length=50)
        train_dataset, test_dataset = processor.create_train_test_split(
            dataset, test_size=0.2, seed=42
        )
        
        assert len(train_dataset) == 80
        assert len(test_dataset) == 20
        assert len(train_dataset) + len(test_dataset) == len(dataset)
    
    def test_get_dataset_info(self):
        """Test getting dataset information."""
        processor = DataProcessor(
            tokenizer_name="bert-base-uncased",
            max_length=512
        )
        
        dataset = processor.create_sample_data(num_samples=10, text_length=50)
        info = processor.get_dataset_info(dataset)
        
        assert "num_examples" in info
        assert "features" in info
        assert "column_names" in info
        assert info["num_examples"] == 10
    
    def test_save_and_load_dataset(self):
        """Test saving and loading dataset."""
        processor = DataProcessor(
            tokenizer_name="bert-base-uncased",
            max_length=512
        )
        
        dataset = processor.create_sample_data(num_samples=5, text_length=50)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_dataset.json"
            processor.save_dataset(dataset, str(filepath), format="json")
            
            assert filepath.exists()
            
            # Load the dataset back
            loaded_dataset = processor.load_dataset_from_file(str(filepath), "json")
            assert len(loaded_dataset) == len(dataset)
    
    def test_create_data_loader(self):
        """Test creating data loader."""
        processor = DataProcessor(
            tokenizer_name="bert-base-uncased",
            max_length=512
        )
        
        dataset = processor.create_sample_data(num_samples=10, text_length=50)
        tokenized_dataset = processor.tokenize_dataset(dataset)
        
        dataloader = processor.create_data_loader(
            tokenized_dataset, batch_size=4, shuffle=True
        )
        
        assert dataloader is not None
        assert dataloader.batch_size == 4
        
        # Test iterating through dataloader
        batch = next(iter(dataloader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 4  # batch_size
    
    def test_invalid_tokenizer_name(self):
        """Test with invalid tokenizer name."""
        with pytest.raises(Exception):
            DataProcessor(
                tokenizer_name="invalid-tokenizer-name",
                max_length=512
            )
    
    def test_unsupported_file_type(self):
        """Test with unsupported file type."""
        processor = DataProcessor(
            tokenizer_name="bert-base-uncased",
            max_length=512
        )
        
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test data")
            f.flush()
            
            with pytest.raises(ValueError):
                processor.load_dataset_from_file(f.name, "xyz")
