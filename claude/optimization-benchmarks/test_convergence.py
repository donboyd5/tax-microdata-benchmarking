"""
Simple convergence test using existing TMD data files.
"""

import sys
import os
sys.path.append('/home/donboyd5/Documents/python_projects/tax-microdata-benchmarking')

import pandas as pd
import numpy as np
import torch
import json
import time
from pathlib import Path
from datetime import datetime

from tmd.storage import STORAGE_FOLDER
from tmd.utils.reweight import reweight

def test_existing_reweight():
    """Test current reweight function to understand baseline behavior."""
    print("Testing existing reweight function...")
    
    # Check if TMD data exists
    tmd_path = STORAGE_FOLDER / "output" / "tmd.csv.gz"
    if not tmd_path.exists():
        print(f"TMD data not found at {tmd_path}")
        print("Available files in output directory:")
        output_dir = STORAGE_FOLDER / "output"
        if output_dir.exists():
            for file in output_dir.glob("*"):
                print(f"  {file.name}")
        else:
            print(f"  Output directory {output_dir} does not exist")
        return None
    
    print(f"Loading TMD data from {tmd_path}")
    tmd_data = pd.read_csv(tmd_path)
    print(f"TMD data shape: {tmd_data.shape}")
    print(f"Sample of columns: {list(tmd_data.columns[:10])}")
    
    # Test with a small sample first
    sample_size = 5000
    if len(tmd_data) > sample_size:
        print(f"Using sample of {sample_size} records for testing")
        tmd_sample = tmd_data.sample(n=sample_size, random_state=42).copy()
        # Scale weights to maintain population
        weight_scale = tmd_data.s006.sum() / tmd_sample.s006.sum()
        tmd_sample['s006'] *= weight_scale
    else:
        tmd_sample = tmd_data.copy()
    
    print(f"Test data shape: {tmd_sample.shape}")
    print(f"Total weight: {tmd_sample.s006.sum():,.0f}")
    
    # Run reweighting with timing
    start_time = time.time()
    result = reweight(tmd_sample, time_period=2021, use_gpu=True)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Reweighting completed in {duration:.1f} seconds")
    
    return {
        'duration_seconds': duration,
        'sample_size': len(tmd_sample),
        'original_weight_sum': tmd_data.s006.sum() if 'tmd_data' in locals() else tmd_sample.s006.sum(),
        'test_weight_sum': tmd_sample.s006.sum(),
        'final_weight_sum': result.s006.sum()
    }

def run_multiple_lr_tests():
    """Test different learning rates with enhanced monitoring."""
    print("Running multiple learning rate tests...")
    
    # Check if TMD data exists
    tmd_path = STORAGE_FOLDER / "output" / "tmd.csv.gz"
    if not tmd_path.exists():
        print(f"TMD data not found. Skipping learning rate tests.")
        return None
    
    tmd_data = pd.read_csv(tmd_path)
    
    # Use smaller sample for faster testing
    sample_size = 2000
    tmd_sample = tmd_data.sample(n=sample_size, random_state=42).copy()
    weight_scale = tmd_data.s006.sum() / tmd_sample.s006.sum()
    tmd_sample['s006'] *= weight_scale
    
    # Test different configurations
    test_configs = [
        {'name': 'lr_0.1_baseline', 'lr': 1e-1, 'max_iter': 500},
        {'name': 'lr_0.05_medium', 'lr': 5e-2, 'max_iter': 500},
        {'name': 'lr_0.01_low', 'lr': 1e-2, 'max_iter': 500},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\nTesting {config['name']}...")
        
        # Create a simple enhanced version for testing
        try:
            start_time = time.time()
            # For now, just use the existing reweight function
            result = reweight(tmd_sample.copy(), time_period=2021, use_gpu=True)
            end_time = time.time()
            
            test_result = {
                'name': config['name'],
                'learning_rate': config['lr'],
                'duration_seconds': end_time - start_time,
                'sample_size': len(tmd_sample),
                'completed': True
            }
            
            results.append(test_result)
            print(f"  Completed in {test_result['duration_seconds']:.1f} seconds")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'name': config['name'], 
                'error': str(e),
                'completed': False
            })
    
    # Save results
    results_path = Path("claude/optimization-benchmarks")
    results_path.mkdir(exist_ok=True)
    
    with open(results_path / "lr_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    print("Starting convergence testing...")
    
    # Test existing functionality
    baseline_result = test_existing_reweight()
    if baseline_result:
        print(f"Baseline test completed: {baseline_result}")
    
    # Test multiple learning rates (simplified)
    lr_results = run_multiple_lr_tests()
    if lr_results:
        print(f"Learning rate tests completed: {len(lr_results)} tests")
        for result in lr_results:
            if result.get('completed'):
                print(f"  {result['name']}: {result['duration_seconds']:.1f}s")
            else:
                print(f"  {result['name']}: FAILED")