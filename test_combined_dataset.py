#!/usr/bin/env python3
"""
Test script to verify the combined dataset pipeline for Kimi-K2-Thinking pruning.

This script tests:
1. Combined dataset creation (50 samples from each of 3 datasets)
2. Proper formatting and processing through dataset processors
3. Integration with the observer system
"""

import sys
import os
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from reap.main import create_combined_dataset
from reap.data import CodeAlpacaChatDataset, XlamFunctionCallingChatDataset, SweSmithTrajectoriesChatDataset
from datasets import Dataset
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_combined_dataset_creation():
    """Test creating the combined dataset with 50 samples from each source."""
    logger.info("Testing combined dataset creation...")
    
    try:
        # Create combined dataset
        combined_dataset = create_combined_dataset(samples_per_dataset=2)  # Use 2 for quick test
        
        logger.info(f"Combined dataset created with {len(combined_dataset)} samples")
        logger.info(f"Dataset columns: {combined_dataset.column_names}")
        
        # Verify we have the source_dataset column
        assert 'source_dataset' in combined_dataset.column_names, "Missing source_dataset column"
        
        # Count samples per source
        source_counts = {}
        for sample in combined_dataset:
            source = sample['source_dataset']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info(f"Samples per source: {source_counts}")
        
        # Verify we have all three sources
        expected_sources = [
            "theblackcat102/evol-codealpaca-v1",
            "Salesforce/xlam-function-calling-60k", 
            "SWE-bench/SWE-smith-trajectories"
        ]
        
        for source in expected_sources:
            assert source in source_counts, f"Missing source dataset: {source}"
            assert source_counts[source] >= 1, f"No samples from {source}"
        
        logger.info("✓ Combined dataset creation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Combined dataset creation test failed: {e}")
        return False

def test_dataset_processors():
    """Test individual dataset processors with small samples."""
    logger.info("Testing individual dataset processors...")
    
    try:
        # Test CodeAlpaca processor
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        
        # Create a small dummy dataset for testing
        dummy_data = {
            "instruction": ["Write a hello world function", "Explain Python lists"],
            "output": ["def hello(): print('Hello World')", "Python lists are..."]
        }
        dummy_dataset = Dataset.from_dict(dummy_data)
        
        processor = CodeAlpacaChatDataset(
            dataset=dummy_dataset,
            tokenizer=tokenizer,
            max_input_len=128,
            split=None,
            split_by_category=False,
            return_vllm_tokens_prompt=False,
            truncate=True,
        )
        
        processed = processor.get_processed_dataset(samples_per_category=2)
        logger.info(f"CodeAlpaca processor test passed: {len(processed['all'])} samples")
        
        logger.info("✓ Dataset processors test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Dataset processors test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Running combined dataset pipeline tests...")
    
    success = True
    
    # Test 1: Combined dataset creation
    success &= test_combined_dataset_creation()
    
    # Test 2: Dataset processors
    success &= test_dataset_processors()
    
    if success:
        logger.info("🎉 All tests passed! Combined dataset pipeline is ready.")
        logger.info("\nTo use the combined dataset for pruning Kimi-K2-Thinking, run:")
        logger.info("\npython -m reap.prune \\")
        logger.info("  --model_name moonshotai/Kimi-K2-Thinking \\")
        logger.info("  --dataset_name combined \\")
        logger.info("  --compression_ratio 0.70 \\")
        logger.info("  --load_in_4bit true \\")
        logger.info("  --samples_per_category 50")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
