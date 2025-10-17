#!/usr/bin/env python3
"""
Test script to check online Hugging Face functionality.
"""

import sys
sys.path.append('src')

def test_online_connection():
    """Test if we can connect to Hugging Face Hub."""
    print("🌐 Testing online Hugging Face connection...")
    
    try:
        from transformers_test import ModelTester, setup_logging
        
        # Setup logging
        logger = setup_logging(level='INFO')
        
        # Test online model loading
        print("📥 Loading BERT model from Hugging Face...")
        tester = ModelTester(
            model_name='bert-base-uncased',
            task_type='classification'
        )
        print("✅ Successfully connected to Hugging Face!")
        
        # Test inference
        print("🧪 Testing inference...")
        result = tester.test_inference("Hello world, this is a test!")
        
        print(f"✅ Online inference test passed!")
        print(f"   Model: {result['model_name']}")
        print(f"   Device: {result['device']}")
        print(f"   Input length: {result['input_length']}")
        print(f"   Predictions shape: {result['predictions'].shape}")
        print(f"   Predicted class: {result['predicted_class']}")
        
        # Test benchmark
        print("⚡ Testing performance benchmark...")
        benchmark = tester.benchmark_performance(
            "This is a benchmark test sentence.",
            num_runs=3,
            warmup_runs=1
        )
        print(f"✅ Benchmark test passed!")
        print(f"   Mean time: {benchmark['mean_time']:.4f}s")
        
        # Get model info
        print("📊 Getting model information...")
        info = tester.get_model_info()
        print(f"✅ Model info retrieved!")
        print(f"   Parameters: {info['num_parameters']:,}")
        print(f"   Model size: {info['model_size_mb']:.2f} MB")
        
        print("\n🎉 All online tests passed! Hugging Face connection is working.")
        return True
        
    except Exception as e:
        print(f"❌ Online test failed: {e}")
        print("\n🔄 Falling back to offline mode...")
        
        try:
            from transformers_test import OfflineModelTester
            offline_tester = OfflineModelTester()
            result = offline_tester.test_inference("Hello world!")
            print(f"✅ Offline inference test passed")
            print(f"   Model: {result['model_name']}")
            print(f"   Predictions: {result['predictions']}")
            return False
        except Exception as offline_e:
            print(f"❌ Offline test also failed: {offline_e}")
            return False

if __name__ == "__main__":
    success = test_online_connection()
    if success:
        print("\n🚀 Your framework is ready for online development!")
        print("You can now use real Hugging Face models.")
    else:
        print("\n⚠️  Using offline mode. Check your internet connection.")
        print("The framework still works with mock models for development.")
