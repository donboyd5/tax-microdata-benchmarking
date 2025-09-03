#!/usr/bin/env python3
"""
Correctly test that the pandas fix produces identical results.
"""

import pandas as pd
import warnings

def test_with_real_structure():
    """Test with the exact structure from the actual code."""
    
    # Set up test data matching the actual code structure
    last_puf_year = 2021
    LAST_YEAR = 2025  # Smaller for testing
    
    # Create initial growth factors DataFrame
    gfdf_data = {
        'YEAR': [2015, 2016, 2017, 2018, 2019, 2020, 2021],
        'AWAGE': [1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12],
        'ASCHCI': [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06],
    }
    gfdf_original = pd.DataFrame(gfdf_data)
    gfdf_fixed = gfdf_original.copy()
    
    print("Initial DataFrame (last few rows):")
    print(gfdf_original.tail(3))
    
    # Test original method (with warning suppression for cleaner output)
    print("\n" + "="*50)
    print("Testing ORIGINAL method (chained assignment):")
    if LAST_YEAR > last_puf_year:
        last_row = gfdf_original.iloc[-1, :].copy()
        num_rows = LAST_YEAR - last_puf_year
        added = pd.DataFrame([last_row] * num_rows, columns=gfdf_original.columns)
        
        # Original chained assignment
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for idx in range(0, num_rows):
                added.YEAR.iat[idx] = last_puf_year + idx + 1
        
        gfdf_original = pd.concat([gfdf_original, added], ignore_index=True)
    
    print(f"Final DataFrame shape: {gfdf_original.shape}")
    print("Last few rows:")
    print(gfdf_original.tail(5))
    
    # Test fixed method
    print("\n" + "="*50)
    print("Testing FIXED method (.loc):")
    if LAST_YEAR > last_puf_year:
        last_row = gfdf_fixed.iloc[-1, :].copy()
        num_rows = LAST_YEAR - last_puf_year
        added = pd.DataFrame([last_row] * num_rows, columns=gfdf_fixed.columns)
        
        # Fixed version using .loc
        for idx in range(0, num_rows):
            added.loc[idx, 'YEAR'] = last_puf_year + idx + 1
        
        gfdf_fixed = pd.concat([gfdf_fixed, added], ignore_index=True)
    
    print(f"Final DataFrame shape: {gfdf_fixed.shape}")
    print("Last few rows:")
    print(gfdf_fixed.tail(5))
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON:")
    print("="*50)
    
    are_identical = gfdf_original.equals(gfdf_fixed)
    print(f"✅ DataFrames are identical: {are_identical}")
    
    if are_identical:
        print("\nVerification:")
        print(f"  Both have {len(gfdf_fixed)} rows")
        print(f"  Both end at year {gfdf_fixed['YEAR'].iloc[-1]}")
        print(f"  Years are sequential in both: {all(gfdf_fixed['YEAR'].diff()[1:] == 1)}")
        
        # Double-check the YEAR column specifically
        year_check = (gfdf_original['YEAR'] == gfdf_fixed['YEAR']).all()
        print(f"  YEAR columns match: {year_check}")
        
        # Check that the added years are correct
        added_years_original = gfdf_original['YEAR'].iloc[7:].tolist()
        added_years_fixed = gfdf_fixed['YEAR'].iloc[7:].tolist()
        expected_years = [2022.0, 2023.0, 2024.0, 2025.0]
        
        print(f"\nAdded years (original): {added_years_original}")
        print(f"Added years (fixed):    {added_years_fixed}")
        print(f"Expected years:         {expected_years}")
        print(f"Match expected: {added_years_original == expected_years and added_years_fixed == expected_years}")
    
    return are_identical

def main():
    print("Testing pandas fix for create_taxcalc_growth_factors.py")
    print("="*50)
    
    success = test_with_real_structure()
    
    print("\n" + "="*50)
    if success:
        print("✅ CONCLUSION: The fix is SAFE!")
        print("   Using .loc[idx, 'YEAR'] produces identical results")
        print("   to the original .YEAR.iat[idx] method.")
    else:
        print("❌ WARNING: The methods produce different results!")
        print("   Further investigation needed.")

if __name__ == "__main__":
    main()