"""
Model testing utilities for Hugging Face Transformers.
"""

import torch
import logging
from typing import Dict, List, Optional, Union, Any
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)


class ModelTester:
    """
    A comprehensive testing framework for Hugging Face Transformers models.
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "classification",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the ModelTester.
        
        Args:
            model_name: Name or path of the model to test
            task_type: Type of task (classification, token_classification, qa, etc.)
            device: Device to run on (auto-detected if None)
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.task_type = task_type
        self.device = device or self._get_device()
        self.cache_dir = cache_dir
        
        self.tokenizer = None
        self.model = None
        self.config = None
        
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
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load config
            self.config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load model based on task type
            if self.task_type == "classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            elif self.task_type == "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            elif self.task_type == "qa":
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def test_inference(
        self, 
        text: Union[str, List[str]], 
        max_length: int = 512,
        return_tensors: str = "pt"
    ) -> Dict[str, Any]:
        """
        Test model inference on given text.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            return_tensors: Format of returned tensors
            
        Returns:
            Dictionary containing model outputs and metadata
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors=return_tensors
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process outputs
            result = {
                "input_text": text,
                "model_name": self.model_name,
                "device": self.device,
                "task_type": self.task_type,
                "outputs": outputs,
                "input_length": inputs["input_ids"].shape[1]
            }
            
            # Add task-specific processing
            if self.task_type == "classification":
                result["predictions"] = torch.softmax(outputs.logits, dim=-1)
                result["predicted_class"] = torch.argmax(outputs.logits, dim=-1)
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def test_batch_inference(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Test batch inference on multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Size of each batch
            max_length: Maximum sequence length
            
        Returns:
            List of inference results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_result = self.test_inference(batch_texts, max_length)
            results.append(batch_result)
        
        return results
    
    def benchmark_performance(
        self,
        text: str,
        num_runs: int = 10,
        warmup_runs: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            text: Input text for benchmarking
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Performance metrics
        """
        import time
        
        # Warmup runs
        for _ in range(warmup_runs):
            self.test_inference(text)
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.test_inference(text)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
            "runs": num_runs
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "device": self.device,
            "config": self.config.to_dict() if self.config else None,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024),
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "max_position_embeddings": self.config.max_position_embeddings if self.config else None
        }
    
    def save_test_results(self, results: Dict[str, Any], filepath: str):
        """Save test results to file."""
        import json
        
        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Test results saved to {filepath}")
    
    def _make_serializable(self, obj):
        """Convert tensors and other non-serializable objects to serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
