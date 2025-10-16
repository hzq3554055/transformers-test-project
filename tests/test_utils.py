"""
Tests for utility functions.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from transformers_test.utils import (
    setup_logging,
    get_device,
    set_seed,
    load_config,
    save_config,
    create_experiment_dir,
    format_time,
    get_model_size,
    count_tokens,
    batch_texts,
    ensure_dir,
    load_json,
    save_json,
    get_gpu_memory_info,
    create_sample_texts
)
import torch
import numpy as np


class TestUtils:
    """Test cases for utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging(level="INFO")
        assert logger is not None
        assert logger.level == 20  # INFO level
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device in ["cpu", "cuda", "mps"]
        
        # Test with specific device
        device = get_device("cpu")
        assert device == "cpu"
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seed is set
        random1 = np.random.random()
        set_seed(42)
        random2 = np.random.random()
        
        # Should be the same with same seed
        assert random1 == random2
    
    def test_load_and_save_config(self):
        """Test loading and saving configuration."""
        config = {
            "model": {"name": "bert-base-uncased"},
            "training": {"epochs": 3}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON
            json_path = Path(temp_dir) / "config.json"
            save_config(config, str(json_path))
            loaded_config = load_config(str(json_path))
            assert loaded_config == config
            
            # Test YAML
            yaml_path = Path(temp_dir) / "config.yaml"
            save_config(config, str(yaml_path))
            loaded_config = load_config(str(yaml_path))
            assert loaded_config == config
    
    def test_create_experiment_dir(self):
        """Test creating experiment directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = create_experiment_dir(temp_dir, "test_experiment")
            
            assert Path(exp_dir).exists()
            assert "test_experiment" in exp_dir
            assert Path(exp_dir).is_dir()
    
    def test_format_time(self):
        """Test time formatting."""
        assert format_time(30) == "30.00s"
        assert format_time(90) == "1.50m"
        assert format_time(7200) == "2.00h"
    
    def test_get_model_size(self):
        """Test getting model size information."""
        # Create a simple model
        model = torch.nn.Linear(10, 5)
        size_info = get_model_size(model)
        
        assert "total_parameters" in size_info
        assert "trainable_parameters" in size_info
        assert "non_trainable_parameters" in size_info
        assert "model_size_mb" in size_info
        assert size_info["total_parameters"] == 55  # 10*5 + 5
        assert size_info["trainable_parameters"] == 55
    
    def test_count_tokens(self):
        """Test counting tokens."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        text = "Hello world"
        
        token_count = count_tokens(text, tokenizer)
        assert token_count > 0
        assert isinstance(token_count, int)
    
    def test_batch_texts(self):
        """Test batching texts."""
        texts = ["text1", "text2", "text3", "text4", "text5"]
        batches = batch_texts(texts, batch_size=2)
        
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1
    
    def test_ensure_dir(self):
        """Test ensuring directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_directory"
            ensure_dir(str(new_dir))
            
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_load_and_save_json(self):
        """Test loading and saving JSON."""
        data = {"key": "value", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_json(data, f.name)
            loaded_data = load_json(f.name)
            assert loaded_data == data
    
    def test_get_gpu_memory_info(self):
        """Test getting GPU memory information."""
        gpu_info = get_gpu_memory_info()
        
        assert "available" in gpu_info
        assert isinstance(gpu_info["available"], bool)
        
        if gpu_info["available"]:
            assert "total_memory_gb" in gpu_info
            assert "allocated_memory_gb" in gpu_info
            assert "cached_memory_gb" in gpu_info
            assert "free_memory_gb" in gpu_info
    
    def test_create_sample_texts(self):
        """Test creating sample texts."""
        texts = create_sample_texts(num_samples=5, text_length=20)
        
        assert len(texts) == 5
        assert all(len(text) == 20 for text in texts)
        assert all(isinstance(text, str) for text in texts)
    
    def test_invalid_config_file(self):
        """Test with invalid config file."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"invalid content")
            f.flush()
            
            with pytest.raises(ValueError):
                load_config(f.name)
    
    def test_nonexistent_config_file(self):
        """Test with non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.json")
