"""
Training utilities for transformers testing.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from transformers.trainer_utils import EvalPrediction
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Training utilities for transformers models.
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "classification",
        num_labels: int = 2,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_name: Name of the model to train
            task_type: Type of task (classification, token_classification, qa)
            num_labels: Number of labels for classification
            device: Device to train on
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        self.device = device or self._get_device()
        self.cache_dir = cache_dir
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        self._load_model()
    
    def _get_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model for training: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load model
            if self.task_type == "classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=self.num_labels,
                    cache_dir=self.cache_dir
                )
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
            
            # Move to device
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./results",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 500,
        save_total_limit: int = 2,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        early_stopping_patience: int = 3,
        **kwargs
    ) -> Trainer:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Output directory for checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Training batch size
            per_device_eval_batch_size: Evaluation batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency
            eval_steps: Evaluation frequency
            save_steps: Save frequency
            save_total_limit: Maximum number of checkpoints to keep
            load_best_model_at_end: Whether to load best model at end
            metric_for_best_model: Metric for model selection
            greater_is_better: Whether higher metric is better
            early_stopping_patience: Early stopping patience
            **kwargs: Additional training arguments
            
        Returns:
            Trained trainer
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                logging_steps=logging_steps,
                eval_steps=eval_steps,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                load_best_model_at_end=load_best_model_at_end,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=greater_is_better,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                logging_dir=f"{output_dir}/logs",
                report_to=None,  # Disable wandb/tensorboard for now
                **kwargs
            )
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics if eval_dataset else None,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if eval_dataset else None
            )
            
            # Train the model
            logger.info("Starting training...")
            train_result = self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Training completed. Results: {train_result}")
            return self.trainer
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for evaluation.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        
        if self.task_type == "classification":
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        else:
            return {}
    
    def evaluate(
        self,
        eval_dataset,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_dataset: Evaluation dataset
            metric_key_prefix: Prefix for metric keys
            
        Returns:
            Evaluation metrics
        """
        try:
            if self.trainer is None:
                raise ValueError("Model must be trained first")
            
            eval_result = self.trainer.evaluate(eval_dataset)
            
            logger.info(f"Evaluation results: {eval_result}")
            return eval_result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def predict(
        self,
        test_dataset,
        return_predictions: bool = True
    ) -> Union[Tuple, np.ndarray]:
        """
        Make predictions on test dataset.
        
        Args:
            test_dataset: Test dataset
            return_predictions: Whether to return predictions
            
        Returns:
            Predictions or tuple of (predictions, labels)
        """
        try:
            if self.trainer is None:
                raise ValueError("Model must be trained first")
            
            predictions = self.trainer.predict(test_dataset)
            
            if return_predictions:
                return predictions.predictions
            else:
                return predictions.predictions, predictions.label_ids
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def save_model(self, output_dir: str):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model
        """
        try:
            if self.trainer is None:
                raise ValueError("Model must be trained first")
            
            os.makedirs(output_dir, exist_ok=True)
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                cache_dir=self.cache_dir
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=self.cache_dir
            )
            
            self.model.to(self.device)
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training setup."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "num_labels": self.num_labels,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
