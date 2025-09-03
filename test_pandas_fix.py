#!/usr/bin/env python3
"""
Test script to verify that the pandas fix produces identical results.
This tests that changing from chained assignment to .loc produces the same output.
"""

import pandas as pd
import numpy as np

def test_original_method():
    """Simulate the original code using chained assignment."""
    # Create a sample DataFrame similar to the one in the code
    last_puf_year = 2021
    LAST_YEAR = 2074
    
    # Simulate some growth factors data
    sample_data = {
        'YEAR': [2021],
        'AWAGE': [1.0],
        'ASCHCI': [1.0],
        'ASCHEI': [1.0],
        'AINTS': [1.0],
        'ADIVS': [1.0]
    }
    gfdf = pd.DataFrame(sample_data)
    
    # Original code logic
    last_row = gfdf.iloc[-1, :].copy()
    num_rows = LAST_YEAR - last_puf_year
    added = pd.DataFrame([last_row] * num_rows, columns=gfdf.columns)
    
    # Original chained assignment (this will show warning)
    for idx in range(0, num_rows):
        added.YEAR.iat[idx] = last_puf_year + idx + 1
    
    result = pd.concat([gfdf, added], ignore_index=True)
    return result

def test_fixed_method():
    """Simulate the fixed code using .loc."""
    # Create a sample DataFrame similar to the one in the code
    last_puf_year = 2021
    LAST_YEAR = 2074
    
    # Simulate some growth factors data
    sample_data = {
        'YEAR': [2021],
        'AWAGE': [1.0],
        'ASCHCI': [1.0],
        'ASCHEI': [1.0],
        'AINTS': [1.0],
        'ADIVS': [1.0]
    }
    gfdf = pd.DataFrame(sample_data)
    
    # Original code logic
    last_row = gfdf.iloc[-1, :].copy()
    num_rows = LAST_YEAR - last_puf_year
    added = pd.DataFrame([last_row] * num_rows, columns=gfdf.columns)
    
    # Fixed version using .loc
    for idx in range(0, num_rows):
        added.loc[idx, 'YEAR'] = last_puf_year + idx + 1
    
    result = pd.concat([gfdf, added], ignore_index=True)
    return result

def main():
    print("Testing pandas chained assignment fix...")
    print("=" * 50)
    
    # Run both methods
    print("\nRunning original method (may show warning)...")
    original_result = test_original_method()
    
    print("\nRunning fixed method...")
    fixed_result = test_fixed_method()
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS:")
    print("=" * 50)
    
    # Check if DataFrames are identical
    are_identical = original_result.equals(fixed_result)
    print(f"DataFrames are identical: {are_identical}")
    
    if not are_identical:
        print("\nDifferences found!")
        print("Shape comparison:")
        print(f"  Original: {original_result.shape}")
        print(f"  Fixed: {fixed_result.shape}")
        
        # Check column by column
        for col in original_result.columns:
            if not original_result[col].equals(fixed_result[col]):
                print(f"\nColumn '{col}' differs:")
                print(f"  Original first 5: {original_result[col].head().values}")
                print(f"  Fixed first 5: {fixed_result[col].head().values}")
    else:
        print("\n✅ SUCCESS: Both methods produce identical results!")
        
        # Show sample of the data to verify it looks correct
        print("\nSample of resulting data (first 5 and last 5 rows):")
        print("\nFirst 5 rows:")
        print(fixed_result.head())
        print("\nLast 5 rows:")
        print(fixed_result.tail())
        
        # Verify YEAR column specifically
        print("\nYEAR column verification:")
        print(f"  First year: {fixed_result['YEAR'].iloc[0]}")
        print(f"  Last year: {fixed_result['YEAR'].iloc[-1]}")
        print(f"  Total rows: {len(fixed_result)}")
        print(f"  Years are sequential: {all(fixed_result['YEAR'].diff()[1:] == 1)}")

if __name__ == "__main__":
    main()