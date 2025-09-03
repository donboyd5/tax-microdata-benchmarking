#!/usr/bin/env python3
"""
Debug why the two methods produce different results.
"""

import pandas as pd
import numpy as np

def debug_issue():
    """Debug the issue with the different approaches."""
    
    # Simulate the data
    last_row_data = {'YEAR': 2021, 'AWAGE': 1.0, 'ASCHCI': 1.0}
    last_row = pd.Series(last_row_data)
    num_rows = 3  # Small number for debugging
    
    print("Creating DataFrame with repeated rows...")
    added = pd.DataFrame([last_row] * num_rows, columns=last_row.index)
    print(f"Initial 'added' DataFrame shape: {added.shape}")
    print(f"Initial 'added' DataFrame:\n{added}")
    print(f"DataFrame index: {added.index.tolist()}")
    
    print("\n" + "="*50)
    print("Testing original method (chained assignment):")
    added1 = added.copy()
    for idx in range(0, num_rows):
        print(f"  Setting added1.YEAR.iat[{idx}] = {2021 + idx + 1}")
        added1.YEAR.iat[idx] = 2021 + idx + 1
    print(f"Result:\n{added1}")
    
    print("\n" + "="*50)
    print("Testing fixed method (.loc):")
    added2 = added.copy()
    for idx in range(0, num_rows):
        print(f"  Setting added2.loc[{idx}, 'YEAR'] = {2021 + idx + 1}")
        added2.loc[idx, 'YEAR'] = 2021 + idx + 1
    print(f"Result:\n{added2}")
    
    print("\n" + "="*50)
    print("Are they equal?", added1.equals(added2))
    
    # Let's also check if the index is reset properly
    print("\nChecking if reset_index helps:")
    added3 = pd.DataFrame([last_row] * num_rows, columns=last_row.index)
    added3 = added3.reset_index(drop=True)
    print(f"After reset_index, DataFrame index: {added3.index.tolist()}")
    for idx in range(0, num_rows):
        added3.loc[idx, 'YEAR'] = 2021 + idx + 1
    print(f"Result with reset_index:\n{added3}")
    
    print("\nComparing with original method result:")
    print("Are they equal?", added1.equals(added3))

if __name__ == "__main__":
    debug_issue()