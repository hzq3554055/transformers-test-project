"""
Offline testing utilities for when internet connection is not available.
"""

import torch
import logging
from typing import Dict, List, Optional, Union, Any
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np

logger = logging.getLogger(__name__)


class OfflineModelTester:
    """
    Offline model tester that works without internet connection.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the offline model tester.
        
        Args:
            device: Device to run on (auto-detected if None)
        """
        self.device = device or self._get_device()
        self.model = None
        self.tokenizer = None
        
    def _get_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def create_mock_model(self, vocab_size: int = 30522, hidden_size: int = 768):
        """
        Create a mock transformer model for testing.
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden layer size
        """
        try:
            # Create a simple mock model
            class MockModel(torch.nn.Module):
                def __init__(self, vocab_size, hidden_size):
                    super().__init__()
                    self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
                    self.transformer = torch.nn.TransformerEncoder(
                        torch.nn.TransformerEncoderLayer(hidden_size, nhead=8),
                        num_layers=6
                    )
                    self.classifier = torch.nn.Linear(hidden_size, 2)
                    
                def forward(self, input_ids, attention_mask=None):
                    x = self.embedding(input_ids)
                    x = self.transformer(x)
                    # Global average pooling
                    x = x.mean(dim=1)
                    return self.classifier(x)
            
            self.model = MockModel(vocab_size, hidden_size)
            self.model.to(self.device)
            self.model.eval()
            
            # Create a simple tokenizer
            class MockTokenizer:
                def __init__(self, vocab_size):
                    self.vocab_size = vocab_size
                    self.pad_token_id = 0
                    self.cls_token_id = 1
                    self.sep_token_id = 2
                    
                def encode(self, text, max_length=512, padding=True, truncation=True):
                    # Simple word-based tokenization
                    words = text.lower().split()
                    tokens = [self.cls_token_id]
                    for word in words[:max_length-2]:
                        # Simple hash-based token ID
                        token_id = hash(word) % (self.vocab_size - 3) + 3
                        tokens.append(token_id)
                    tokens.append(self.sep_token_id)
                    
                    if padding and len(tokens) < max_length:
                        tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
                    elif truncation and len(tokens) > max_length:
                        tokens = tokens[:max_length]
                        
                    return tokens
                    
                def __call__(self, text, max_length=512, padding=True, truncation=True, return_tensors="pt"):
                    if isinstance(text, str):
                        tokens = self.encode(text, max_length, padding, truncation)
                        return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.tensor([[1] * len(tokens)])}
                    else:
                        # Batch processing
                        batch_tokens = []
                        batch_masks = []
                        for t in text:
                            tokens = self.encode(t, max_length, padding, truncation)
                            batch_tokens.append(tokens)
                            batch_masks.append([1] * len(tokens))
                        return {
                            "input_ids": torch.tensor(batch_tokens),
                            "attention_mask": torch.tensor(batch_masks)
                        }
            
            self.tokenizer = MockTokenizer(vocab_size)
            
            logger.info(f"Mock model created successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to create mock model: {e}")
            raise
    
    def test_inference(self, text: Union[str, List[str]], max_length: int = 512) -> Dict[str, Any]:
        """
        Test model inference on given text.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing model outputs and metadata
        """
        if self.model is None:
            self.create_mock_model()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process outputs
            result = {
                "input_text": text,
                "model_name": "mock-model",
                "device": self.device,
                "task_type": "classification",
                "outputs": outputs,
                "input_length": inputs["input_ids"].shape[1]
            }
            
            # Add predictions
            result["predictions"] = torch.softmax(outputs, dim=-1)
            result["predicted_class"] = torch.argmax(outputs, dim=-1)
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def benchmark_performance(self, text: str, num_runs: int = 10, warmup_runs: int = 3) -> Dict[str, float]:
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
        
        if self.model is None:
            self.create_mock_model()
        
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
        if self.model is None:
            self.create_mock_model()
            
        return {
            "model_name": "mock-model",
            "task_type": "classification",
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024),
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None
        }


def create_sample_data(num_samples: int = 10, text_length: int = 50) -> List[Dict[str, Any]]:
    """
    Create sample data for offline testing.
    
    Args:
        num_samples: Number of samples to create
        text_length: Length of each text sample
        
    Returns:
        List of sample data dictionaries
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
    
    return data


def run_offline_demo():
    """Run a complete offline demo."""
    print("🚀 Running Offline Transformers Testing Demo")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize offline tester
    tester = OfflineModelTester()
    
    # Test inference
    print("\n📝 Testing Model Inference...")
    text = "This is a test sentence for offline testing."
    result = tester.test_inference(text)
    
    print(f"Input: {result['input_text']}")
    print(f"Model: {result['model_name']}")
    print(f"Device: {result['device']}")
    print(f"Predictions: {result['predictions']}")
    print(f"Predicted class: {result['predicted_class']}")
    
    # Benchmark performance
    print("\n⚡ Benchmarking Performance...")
    benchmark = tester.benchmark_performance(text, num_runs=5, warmup_runs=2)
    
    print(f"Mean time: {benchmark['mean_time']:.4f}s")
    print(f"Min time: {benchmark['min_time']:.4f}s")
    print(f"Max time: {benchmark['max_time']:.4f}s")
    
    # Get model info
    print("\n📊 Model Information...")
    info = tester.get_model_info()
    print(f"Parameters: {info['num_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    
    # Test batch inference
    print("\n🔄 Testing Batch Inference...")
    texts = ["First test sentence.", "Second test sentence.", "Third test sentence."]
    batch_result = tester.test_inference(texts)
    print(f"Batch processed: {len(texts)} texts")
    
    # Create sample data
    print("\n📊 Creating Sample Data...")
    sample_data = create_sample_data(num_samples=5, text_length=30)
    print(f"Created {len(sample_data)} sample data points")
    
    print("\n✅ Offline demo completed successfully!")
    print("\nThis demonstrates that the framework works even without internet connection.")
    print("You can use this for development and testing when offline.")


if __name__ == "__main__":
    run_offline_demo()
