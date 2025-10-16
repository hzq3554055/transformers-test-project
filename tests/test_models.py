"""
Tests for model testing utilities.
"""

import pytest
import torch
from transformers_test.models import ModelTester


class TestModelTester:
    """Test cases for ModelTester class."""
    
    def test_initialization(self):
        """Test ModelTester initialization."""
        tester = ModelTester(
            model_name="bert-base-uncased",
            task_type="classification"
        )
        
        assert tester.model_name == "bert-base-uncased"
        assert tester.task_type == "classification"
        assert tester.model is not None
        assert tester.tokenizer is not None
    
    def test_inference(self):
        """Test model inference."""
        tester = ModelTester(
            model_name="bert-base-uncased",
            task_type="classification"
        )
        
        text = "Hello world"
        result = tester.test_inference(text)
        
        assert "input_text" in result
        assert "outputs" in result
        assert "predictions" in result
        assert result["input_text"] == text
    
    def test_batch_inference(self):
        """Test batch inference."""
        tester = ModelTester(
            model_name="bert-base-uncased",
            task_type="classification"
        )
        
        texts = ["Hello world", "Goodbye world"]
        results = tester.test_batch_inference(texts, batch_size=2)
        
        assert len(results) == 1  # Single batch
        assert "input_text" in results[0]
    
    def test_benchmark_performance(self):
        """Test performance benchmarking."""
        tester = ModelTester(
            model_name="bert-base-uncased",
            task_type="classification"
        )
        
        text = "Hello world"
        results = tester.benchmark_performance(text, num_runs=3, warmup_runs=1)
        
        assert "mean_time" in results
        assert "min_time" in results
        assert "max_time" in results
        assert results["runs"] == 3
    
    def test_get_model_info(self):
        """Test getting model information."""
        tester = ModelTester(
            model_name="bert-base-uncased",
            task_type="classification"
        )
        
        info = tester.get_model_info()
        
        assert "model_name" in info
        assert "task_type" in info
        assert "device" in info
        assert "num_parameters" in info
        assert info["model_name"] == "bert-base-uncased"
        assert info["task_type"] == "classification"
    
    def test_invalid_model_name(self):
        """Test with invalid model name."""
        with pytest.raises(Exception):
            ModelTester(
                model_name="invalid-model-name",
                task_type="classification"
            )
    
    def test_unsupported_task_type(self):
        """Test with unsupported task type."""
        with pytest.raises(Exception):
            ModelTester(
                model_name="bert-base-uncased",
                task_type="unsupported_task"
            )
