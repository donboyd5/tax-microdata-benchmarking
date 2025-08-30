"""
Early Stopping Analysis for National Weight Optimization

This script analyzes convergence patterns to determine optimal early stopping 
criteria based on a 0.5% target tolerance for all SOI statistics.
"""

import sys
import os
sys.path.append('/home/donboyd5/Documents/python_projects/tax-microdata-benchmarking')

import pandas as pd
import numpy as np
import torch
import json
import time
# import matplotlib.pyplot as plt  # Skip plotting for now
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from tmd.storage import STORAGE_FOLDER
from tmd.utils.soi_replication import tc_to_soi
from tmd.imputation_assumptions import (
    REWEIGHT_MULTIPLIER_MIN,
    REWEIGHT_MULTIPLIER_MAX,
    REWEIGHT_DEVIATION_PENALTY,
)


def analyze_convergence_pattern(
    flat_file: pd.DataFrame,
    time_period: int = 2021,
    target_tolerance: float = 0.005,  # 0.5%
    max_iterations: int = 2000,
    analysis_frequency: int = 10,  # Check convergence every N iterations
    use_gpu: bool = True,
):
    """
    Analyze convergence patterns to determine early stopping criteria.
    
    Args:
        flat_file: DataFrame to reweight
        time_period: Year to target
        target_tolerance: Acceptable relative error (0.005 = 0.5%)
        max_iterations: Maximum iterations to analyze
        analysis_frequency: How often to check convergence
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Dict with convergence analysis results
    """
    
    print(f"Analyzing convergence patterns with {target_tolerance*100:.1f}% tolerance")
    
    # Load SOI targets
    targets = pd.read_csv(STORAGE_FOLDER / "input" / "soi.csv")
    if time_period not in targets.Year.unique():
        raise ValueError(f"Year {time_period} not in targets.")

    # Build loss matrix (same as production reweight.py)
    def build_loss_matrix(df):
        loss_matrix = pd.DataFrame()
        df = tc_to_soi(df, time_period)
        agi = df["adjusted_gross_income"].values
        filer = df["is_tax_filer"].values
        targets_array = []
        soi_subset = targets[targets.Year == time_period]
        
        agi_level_targeted_variables = [
            "adjusted_gross_income", "count", "employment_income",
            "business_net_profits", "capital_gains_gross", "ordinary_dividends",
            "partnership_and_s_corp_income", "qualified_dividends",
            "taxable_interest_income", "total_pension_income", "total_social_security",
        ]
        aggregate_level_targeted_variables = [
            "business_net_losses", "capital_gains_distributions", "capital_gains_losses",
            "estate_income", "estate_losses", "exempt_interest", "ira_distributions",
            "partnership_and_s_corp_losses", "rent_and_royalty_net_income",
            "rent_and_royalty_net_losses", "taxable_pension_income",
            "taxable_social_security", "unemployment_compensation",
        ]
        aggregate_level_targeted_variables = [
            variable for variable in aggregate_level_targeted_variables
            if variable in df.columns
        ]
        
        soi_subset = soi_subset[
            soi_subset.Variable.isin(agi_level_targeted_variables)
            & (
                (soi_subset["AGI lower bound"] != -np.inf)
                | (soi_subset["AGI upper bound"] != np.inf)
            )
            | (
                soi_subset.Variable.isin(aggregate_level_targeted_variables)
                & (soi_subset["AGI lower bound"] == -np.inf)
                & (soi_subset["AGI upper bound"] == np.inf)
            )
        ]
        
        for _, row in soi_subset.iterrows():
            if row["Taxable only"]:
                continue
                
            mask = (
                (agi >= row["AGI lower bound"])
                * (agi < row["AGI upper bound"])
                * filer
            ) > 0

            if row["Filing status"] == "Single":
                mask *= df["filing_status"].values == "SINGLE"
            elif row["Filing status"] == "Married Filing Jointly/Surviving Spouse":
                mask *= df["filing_status"].values == "JOINT"
            elif row["Filing status"] == "Head of Household":
                mask *= df["filing_status"].values == "HEAD_OF_HOUSEHOLD"
            elif row["Filing status"] == "Married Filing Separately":
                mask *= df["filing_status"].values == "SEPARATE"

            values = df[row["Variable"]].values
            if row["Count"]:
                values = (values > 0).astype(float)

            def fmt(x):
                if x == -np.inf: return "-inf"
                if x == np.inf: return "inf"
                if x < 1e3: return f"{x:.0f}"
                if x < 1e6: return f"{x/1e3:.0f}k"
                if x < 1e9: return f"{x/1e6:.0f}m"
                return f"{x/1e9:.1f}bn"

            agi_range_label = f"{fmt(row['AGI lower bound'])}-{fmt(row['AGI upper bound'])}"
            taxable_label = "taxable" if row["Taxable only"] else "all" + " returns"
            filing_status_label = row["Filing status"]
            variable_label = row["Variable"].replace("_", " ")

            if row["Count"] and not row["Variable"] == "count":
                label = f"{variable_label}/count/AGI in {agi_range_label}/{taxable_label}/{filing_status_label}"
            elif row["Variable"] == "count":
                label = f"{variable_label}/count/AGI in {agi_range_label}/{taxable_label}/{filing_status_label}"
            else:
                label = f"{variable_label}/total/AGI in {agi_range_label}/{taxable_label}/{filing_status_label}"

            if label not in loss_matrix.columns:
                loss_matrix[label] = mask * values
                targets_array.append(row["Value"])

        return loss_matrix.copy(), np.array(targets_array)

    # Setup GPU/CPU
    gpu_available = torch.cuda.is_available()
    use_gpu_actual = use_gpu and gpu_available
    device = torch.device("cuda" if use_gpu_actual else "cpu")
    
    if use_gpu_actual:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")

    # Set random seeds
    torch.manual_seed(65748392)
    torch.cuda.manual_seed_all(65748392)

    # Create tensors
    weights = torch.tensor(flat_file.s006.values, dtype=torch.float32, device=device)
    weight_multiplier = torch.tensor(
        np.ones_like(flat_file.s006.values),
        dtype=torch.float32, device=device, requires_grad=True
    )
    original_weights = weights.clone()
    output_matrix, target_array = build_loss_matrix(flat_file)
    
    print(f"Analyzing {len(target_array)} SOI statistics")
    
    output_matrix_tensor = torch.tensor(output_matrix.values, dtype=torch.float32, device=device)
    target_array_tensor = torch.tensor(target_array, dtype=torch.float32, device=device)
    
    # Calculate initial outputs and loss
    outputs = (weights * output_matrix_tensor.T).sum(axis=1)
    original_loss_value = (((outputs + 1) / (target_array_tensor + 1) - 1) ** 2).sum()
    
    # Setup optimizer
    optimizer = torch.optim.Adam([weight_multiplier], lr=1e-1)
    
    # Convergence tracking
    convergence_data = {
        'iteration': [],
        'total_loss': [],
        'max_rel_error': [],
        'mean_rel_error': [],
        'median_rel_error': [],
        'pct_within_tolerance': [],
        'worst_targets': [],
        'early_stop_would_trigger': []
    }
    
    print(f"Starting convergence analysis (checking every {analysis_frequency} iterations)...")
    start_time = time.time()
    
    for i in tqdm(range(max_iterations), desc="Analyzing convergence"):
        optimizer.zero_grad()
        new_weights = weights * (
            torch.clamp(weight_multiplier, min=REWEIGHT_MULTIPLIER_MIN, max=REWEIGHT_MULTIPLIER_MAX)
        )
        outputs = (new_weights * output_matrix_tensor.T).sum(axis=1)
        weight_deviation = (
            (new_weights - original_weights).abs().sum()
            / original_weights.sum()
            * REWEIGHT_DEVIATION_PENALTY
            * original_loss_value
        )
        loss_value = (((outputs + 1) / (target_array_tensor + 1) - 1) ** 2).sum() + weight_deviation
        loss_value.backward()
        optimizer.step()
        
        # Analyze convergence every N iterations
        if i % analysis_frequency == 0:
            with torch.no_grad():
                rel_errors = ((outputs + 1) / (target_array_tensor + 1) - 1).abs()
                max_error = rel_errors.max().item()
                mean_error = rel_errors.mean().item()
                median_error = rel_errors.median().item()
                
                # Calculate percentage within tolerance
                within_tolerance = (rel_errors <= target_tolerance).float().mean().item()
                
                # Find worst performing targets
                worst_indices = torch.topk(rel_errors, k=min(5, len(rel_errors)))[1].cpu().numpy()
                worst_targets = [
                    {
                        'target': output_matrix.columns[idx],
                        'error_pct': rel_errors[idx].item() * 100,
                        'actual': outputs[idx].item(),
                        'target_value': target_array[idx]
                    }
                    for idx in worst_indices
                ]
                
                # Check if early stopping would trigger
                early_stop_trigger = within_tolerance >= 0.95  # 95% of targets within tolerance
                
                # Store data
                convergence_data['iteration'].append(i)
                convergence_data['total_loss'].append(loss_value.item())
                convergence_data['max_rel_error'].append(max_error)
                convergence_data['mean_rel_error'].append(mean_error)
                convergence_data['median_rel_error'].append(median_error)
                convergence_data['pct_within_tolerance'].append(within_tolerance * 100)
                convergence_data['worst_targets'].append(worst_targets)
                convergence_data['early_stop_would_trigger'].append(early_stop_trigger)
    
    end_time = time.time()
    
    # Analysis results
    results = {
        'convergence_data': convergence_data,
        'final_metrics': {
            'max_rel_error_pct': convergence_data['max_rel_error'][-1] * 100,
            'mean_rel_error_pct': convergence_data['mean_rel_error'][-1] * 100,
            'median_rel_error_pct': convergence_data['median_rel_error'][-1] * 100,
            'pct_within_tolerance': convergence_data['pct_within_tolerance'][-1],
            'final_worst_targets': convergence_data['worst_targets'][-1]
        },
        'early_stopping_analysis': analyze_early_stopping_potential(convergence_data, target_tolerance),
        'performance': {
            'total_time_seconds': end_time - start_time,
            'iterations_per_second': max_iterations / (end_time - start_time),
            'target_tolerance_pct': target_tolerance * 100
        }
    }
    
    return results


def analyze_early_stopping_potential(convergence_data, target_tolerance):
    """Analyze when early stopping would be beneficial."""
    
    iterations = np.array(convergence_data['iteration'])
    pct_within_tolerance = np.array(convergence_data['pct_within_tolerance'])
    max_errors = np.array(convergence_data['max_rel_error'])
    
    # Find when we first achieve different thresholds
    thresholds = [90, 95, 98, 99, 99.5]
    threshold_results = {}
    
    for threshold in thresholds:
        first_achieved = np.where(pct_within_tolerance >= threshold)[0]
        if len(first_achieved) > 0:
            first_iter = iterations[first_achieved[0]]
            threshold_results[f'{threshold}pct_targets'] = {
                'first_achieved_iteration': int(first_iter),
                'potential_time_saved_pct': (2000 - first_iter) / 2000 * 100,
                'max_error_at_achievement': max_errors[first_achieved[0]] * 100
            }
        else:
            threshold_results[f'{threshold}pct_targets'] = None
    
    # Find when max error drops below certain levels
    error_thresholds = [0.01, 0.005, 0.002, 0.001]  # 1%, 0.5%, 0.2%, 0.1%
    error_results = {}
    
    for error_threshold in error_thresholds:
        first_below = np.where(max_errors <= error_threshold)[0]
        if len(first_below) > 0:
            first_iter = iterations[first_below[0]]
            error_results[f'max_error_below_{error_threshold*100:.1f}pct'] = {
                'first_achieved_iteration': int(first_iter),
                'potential_time_saved_pct': (2000 - first_iter) / 2000 * 100,
                'pct_within_tolerance_at_achievement': pct_within_tolerance[first_below[0]]
            }
        else:
            error_results[f'max_error_below_{error_threshold*100:.1f}pct'] = None
    
    return {
        'target_percentage_thresholds': threshold_results,
        'max_error_thresholds': error_results,
        'recommended_early_stopping': recommend_early_stopping_strategy(
            threshold_results, error_results, target_tolerance
        )
    }


def recommend_early_stopping_strategy(threshold_results, error_results, target_tolerance):
    """Recommend optimal early stopping strategy based on analysis."""
    
    recommendations = []
    
    # Strategy 1: 95% of targets within tolerance
    if threshold_results.get('95pct_targets'):
        strategy1 = threshold_results['95pct_targets']
        recommendations.append({
            'strategy': '95% of targets within 0.5% tolerance',
            'stop_iteration': strategy1['first_achieved_iteration'],
            'time_savings_pct': strategy1['potential_time_saved_pct'],
            'max_error_pct': strategy1['max_error_at_achievement'],
            'recommended': True if strategy1['potential_time_saved_pct'] > 20 else False,
            'reason': 'Good balance of accuracy and time savings' if strategy1['potential_time_saved_pct'] > 20 else 'Insufficient time savings'
        })
    
    # Strategy 2: Max error below 1%
    if error_results.get('max_error_below_1.0pct'):
        strategy2 = error_results['max_error_below_1.0pct']
        recommendations.append({
            'strategy': 'Maximum error below 1%',
            'stop_iteration': strategy2['first_achieved_iteration'],
            'time_savings_pct': strategy2['potential_time_saved_pct'],
            'pct_within_tolerance': strategy2['pct_within_tolerance_at_achievement'],
            'recommended': True if strategy2['potential_time_saved_pct'] > 15 else False,
            'reason': 'Conservative approach with good accuracy' if strategy2['potential_time_saved_pct'] > 15 else 'Minimal time savings'
        })
    
    # Strategy 3: Max error below 0.5% (matches target tolerance)
    if error_results.get('max_error_below_0.5pct'):
        strategy3 = error_results['max_error_below_0.5pct']
        recommendations.append({
            'strategy': 'Maximum error below 0.5% (target tolerance)',
            'stop_iteration': strategy3['first_achieved_iteration'],
            'time_savings_pct': strategy3['potential_time_saved_pct'],
            'pct_within_tolerance': strategy3['pct_within_tolerance_at_achievement'],
            'recommended': True,
            'reason': 'Matches specified tolerance exactly'
        })
    
    return recommendations


def create_convergence_plots(convergence_data, save_path="claude/optimization-benchmarks"):
    """Create visualization plots of convergence behavior."""
    # Skip plotting for now - matplotlib not available
    print("Skipping plot generation (matplotlib not available)")
    return


def run_early_stopping_analysis():
    """Run the complete early stopping analysis."""
    
    print("Loading TMD data for early stopping analysis...")
    tmd_path = STORAGE_FOLDER / "output" / "tmd.csv.gz"
    
    if not tmd_path.exists():
        print(f"TMD data not found at {tmd_path}")
        return None
    
    tmd_data = pd.read_csv(tmd_path)
    print(f"Loaded TMD data: {tmd_data.shape}")
    
    # Run convergence analysis
    results = analyze_convergence_pattern(
        tmd_data,
        target_tolerance=0.005,  # 0.5%
        max_iterations=2000,
        analysis_frequency=10,
        use_gpu=True
    )
    
    # Create plots
    create_convergence_plots(results['convergence_data'])
    
    # Save detailed results
    results_path = Path("claude/optimization-benchmarks")
    results_path.mkdir(exist_ok=True)
    
    with open(results_path / "early_stopping_analysis.json", 'w') as f:
        # Make results JSON serializable
        json_results = results.copy()
        json_results['convergence_data']['worst_targets'] = [
            [{'target': t['target'], 'error_pct': float(t['error_pct']), 
              'actual': float(t['actual']), 'target_value': float(t['target_value'])}
             for t in targets] 
            for targets in json_results['convergence_data']['worst_targets']
        ]
        json.dump(json_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EARLY STOPPING ANALYSIS SUMMARY")
    print("="*60)
    
    final = results['final_metrics']
    print(f"Final Results (after 2000 iterations):")
    print(f"  Max relative error: {final['max_rel_error_pct']:.2f}%")
    print(f"  Mean relative error: {final['mean_rel_error_pct']:.2f}%") 
    print(f"  Targets within 0.5% tolerance: {final['pct_within_tolerance']:.1f}%")
    
    print(f"\nWorst performing targets:")
    for i, target in enumerate(final['final_worst_targets'][:3]):
        print(f"  {i+1}. {target['target']}: {target['error_pct']:.2f}% error")
    
    print(f"\nEarly Stopping Recommendations:")
    recommendations = results['early_stopping_analysis']['recommended_early_stopping']
    for rec in recommendations:
        status = "✅ RECOMMENDED" if rec.get('recommended') else "❌ NOT RECOMMENDED"
        print(f"  {status}: {rec['strategy']}")
        print(f"    Stop at iteration: {rec['stop_iteration']}")
        print(f"    Time savings: {rec['time_savings_pct']:.1f}%")
        print(f"    Reason: {rec['reason']}")
        print()
    
    return results


if __name__ == "__main__":
    run_early_stopping_analysis()