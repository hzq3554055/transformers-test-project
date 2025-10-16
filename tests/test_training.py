"""
Tests for training utilities.
"""

import pytest
import tempfile
from pathlib import Path
from transformers_test.training import ModelTrainer
from transformers_test.data import DataProcessor


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(
            model_name="bert-base-uncased",
            task_type="classification",
            num_labels=2
        )
        
        assert trainer.model_name == "bert-base-uncased"
        assert trainer.task_type == "classification"
        assert trainer.num_labels == 2
        assert trainer.model is not None
        assert trainer.tokenizer is not None
    
    def test_get_training_info(self):
        """Test getting training information."""
        trainer = ModelTrainer(
            model_name="bert-base-uncased",
            task_type="classification",
            num_labels=2
        )
        
        info = trainer.get_training_info()
        
        assert "model_name" in info
        assert "task_type" in info
        assert "num_labels" in info
        assert "device" in info
        assert "num_parameters" in info
        assert "trainable_parameters" in info
        assert info["model_name"] == "bert-base-uncased"
        assert info["task_type"] == "classification"
        assert info["num_labels"] == 2
    
    def test_compute_metrics(self):
        """Test computing metrics."""
        trainer = ModelTrainer(
            model_name="bert-base-uncased",
            task_type="classification",
            num_labels=2
        )
        
        # Create mock predictions and labels
        import numpy as np
        predictions = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        labels = np.array([0, 1, 0])
        
        eval_pred = type('EvalPrediction', (), {
            'predictions': predictions,
            'label_ids': labels
        })()
        
        metrics = trainer._compute_metrics(eval_pred)
        
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1
    
    def test_unsupported_task_type(self):
        """Test with unsupported task type."""
        with pytest.raises(ValueError):
            ModelTrainer(
                model_name="bert-base-uncased",
                task_type="unsupported_task",
                num_labels=2
            )
    
    def test_invalid_model_name(self):
        """Test with invalid model name."""
        with pytest.raises(Exception):
            ModelTrainer(
                model_name="invalid-model-name",
                task_type="classification",
                num_labels=2
            )
    
    @pytest.mark.slow
    def test_train_with_sample_data(self):
        """Test training with sample data (slow test)."""
        trainer = ModelTrainer(
            model_name="bert-base-uncased",
            task_type="classification",
            num_labels=2
        )
        
        # Create sample data
        data_processor = DataProcessor(
            tokenizer_name="bert-base-uncased",
            max_length=128
        )
        
        dataset = data_processor.create_sample_data(num_samples=20, text_length=50)
        train_dataset, eval_dataset = data_processor.create_train_test_split(
            dataset, test_size=0.3, seed=42
        )
        
        # Tokenize datasets
        train_dataset = data_processor.tokenize_dataset(train_dataset)
        eval_dataset = data_processor.tokenize_dataset(eval_dataset)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train model
            trained_trainer = trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=temp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                logging_steps=1,
                eval_steps=1,
                save_steps=1000  # Don't save during training
            )
            
            assert trained_trainer is not None
            assert Path(temp_dir).exists()
    
    def test_save_and_load_model(self):
        """Test saving and loading model."""
        trainer = ModelTrainer(
            model_name="bert-base-uncased",
            task_type="classification",
            num_labels=2
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            trainer.save_model(temp_dir)
            
            assert Path(temp_dir).exists()
            
            # Create new trainer and load model
            new_trainer = ModelTrainer(
                model_name="bert-base-uncased",
                task_type="classification",
                num_labels=2
            )
            
            new_trainer.load_model(temp_dir)
            
            assert new_trainer.model is not None
            assert new_trainer.tokenizer is not None
