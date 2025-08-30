"""
Enhanced reweight.py with convergence monitoring and optimization testing.
This version adds detailed convergence tracking and parameter experimentation.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
import time
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from tmd.storage import STORAGE_FOLDER
from tmd.utils.soi_replication import tc_to_soi
from tmd.imputation_assumptions import (
    REWEIGHT_MULTIPLIER_MIN,
    REWEIGHT_MULTIPLIER_MAX,
    REWEIGHT_DEVIATION_PENALTY,
)


def enhanced_reweight(
    flat_file: pd.DataFrame,
    time_period: int = 2021,
    weight_multiplier_min: float = REWEIGHT_MULTIPLIER_MIN,
    weight_multiplier_max: float = REWEIGHT_MULTIPLIER_MAX,
    weight_deviation_penalty: float = REWEIGHT_DEVIATION_PENALTY,
    use_gpu: bool = True,
    # New optimization parameters
    learning_rate: float = 1e-1,
    max_iterations: int = 2000,
    logging_frequency: int = 100,
    early_stopping: bool = True,
    early_stopping_patience: int = 100,
    early_stopping_threshold: float = 0.001,  # 0.1% improvement threshold
    benchmark_name: str = "default",
    save_convergence_data: bool = True,
):
    """
    Enhanced reweighting function with convergence monitoring and optimization testing.
    
    Args:
        flat_file: DataFrame to reweight
        time_period: Year to target
        weight_multiplier_min/max: Weight bounds
        weight_deviation_penalty: Penalty for weight changes
        use_gpu: Whether to use GPU acceleration
        learning_rate: Adam optimizer learning rate
        max_iterations: Maximum optimization iterations
        logging_frequency: How often to log metrics
        early_stopping: Whether to use early stopping
        early_stopping_patience: Iterations to wait for improvement
        early_stopping_threshold: Minimum relative improvement required
        benchmark_name: Name for saving results
        save_convergence_data: Whether to save detailed convergence data
    
    Returns:
        Tuple of (reweighted_dataframe, convergence_results)
    """
    
    print(f"...Enhanced reweighting for year {time_period}")
    print(f"...Benchmark: {benchmark_name}")
    print(f"...Learning rate: {learning_rate}")
    print(f"...Max iterations: {max_iterations}")
    print(f"...Early stopping: {early_stopping}")
    
    targets = pd.read_csv(STORAGE_FOLDER / "input" / "soi.csv")

    if time_period not in targets.Year.unique():
        raise ValueError(f"Year {time_period} not in targets.")

    def build_loss_matrix(df):
        loss_matrix = pd.DataFrame()
        df = tc_to_soi(df, time_period)
        agi = df["adjusted_gross_income"].values
        filer = df["is_tax_filer"].values
        targets_array = []
        soi_subset = targets
        soi_subset = soi_subset[soi_subset.Year == time_period]
        agi_level_targeted_variables = [
            "adjusted_gross_income",
            "count",
            "employment_income",
            "business_net_profits",
            "capital_gains_gross",
            "ordinary_dividends",
            "partnership_and_s_corp_income",
            "qualified_dividends",
            "taxable_interest_income",
            "total_pension_income",
            "total_social_security",
        ]
        aggregate_level_targeted_variables = [
            "business_net_losses",
            "capital_gains_distributions",
            "capital_gains_losses",
            "estate_income",
            "estate_losses",
            "exempt_interest",
            "ira_distributions",
            "partnership_and_s_corp_losses",
            "rent_and_royalty_net_income",
            "rent_and_royalty_net_losses",
            "taxable_pension_income",
            "taxable_social_security",
            "unemployment_compensation",
        ]
        aggregate_level_targeted_variables = [
            variable
            for variable in aggregate_level_targeted_variables
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
                continue  # exclude "taxable returns" statistics

            mask = (
                (agi >= row["AGI lower bound"])
                * (agi < row["AGI upper bound"])
                * filer
            ) > 0

            if row["Filing status"] == "Single":
                mask *= df["filing_status"].values == "SINGLE"
            elif (
                row["Filing status"]
                == "Married Filing Jointly/Surviving Spouse"
            ):
                mask *= df["filing_status"].values == "JOINT"
            elif row["Filing status"] == "Head of Household":
                mask *= df["filing_status"].values == "HEAD_OF_HOUSEHOLD"
            elif row["Filing status"] == "Married Filing Separately":
                mask *= df["filing_status"].values == "SEPARATE"

            values = df[row["Variable"]].values

            if row["Count"]:
                values = (values > 0).astype(float)

            def fmt(x):
                if x == -np.inf:
                    return "-inf"
                if x == np.inf:
                    return "inf"
                if x < 1e3:
                    return f"{x:.0f}"
                if x < 1e6:
                    return f"{x/1e3:.0f}k"
                if x < 1e9:
                    return f"{x/1e6:.0f}m"
                return f"{x/1e9:.1f}bn"

            agi_range_label = (
                f"{fmt(row['AGI lower bound'])}-{fmt(row['AGI upper bound'])}"
            )
            taxable_label = (
                "taxable" if row["Taxable only"] else "all" + " returns"
            )
            filing_status_label = row["Filing status"]

            variable_label = row["Variable"].replace("_", " ")

            if row["Count"] and not row["Variable"] == "count":
                label = (
                    f"{variable_label}/count/AGI in "
                    f"{agi_range_label}/{taxable_label}/{filing_status_label}"
                )
            elif row["Variable"] == "count":
                label = (
                    f"{variable_label}/count/AGI in "
                    f"{agi_range_label}/{taxable_label}/{filing_status_label}"
                )
            else:
                label = (
                    f"{variable_label}/total/AGI in "
                    f"{agi_range_label}/{taxable_label}/{filing_status_label}"
                )

            if label not in loss_matrix.columns:
                loss_matrix[label] = mask * values
                targets_array.append(row["Value"])

        return loss_matrix.copy(), np.array(targets_array)

    # GPU Detection and Device Selection
    gpu_available = torch.cuda.is_available()
    use_gpu_actual = use_gpu and gpu_available

    device = torch.device("cuda" if use_gpu_actual else "cpu")

    if use_gpu_actual:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"...GPU acceleration enabled: {gpu_name} ({gpu_memory:.1f} GB)")
    elif use_gpu and not gpu_available:
        print("...GPU requested but not available, using CPU")
    elif not use_gpu and gpu_available:
        print("...GPU available but disabled by user, using CPU")
    else:
        print("...GPU not available, using CPU")

    # Set random seeds for reproducibility
    rng_seed = 65748392
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)

    # Create tensors directly on the selected device
    weights = torch.tensor(
        flat_file.s006.values, dtype=torch.float32, device=device
    )
    weight_multiplier = torch.tensor(
        np.ones_like(flat_file.s006.values),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    original_weights = weights.clone()
    output_matrix, target_array = build_loss_matrix(flat_file)

    print(f"Targeting {len(target_array)} SOI statistics")
    
    output_matrix_tensor = torch.tensor(
        output_matrix.values, dtype=torch.float32, device=device
    )
    target_array = torch.tensor(
        target_array, dtype=torch.float32, device=device
    )

    outputs = (weights * output_matrix_tensor.T).sum(axis=1)
    original_loss_value = (((outputs + 1) / (target_array + 1) - 1) ** 2).sum()

    # Create optimizer
    optimizer = torch.optim.Adam([weight_multiplier], lr=learning_rate)

    # Setup TensorBoard logging
    log_dir = STORAGE_FOLDER / "output" / "reweighting" / f"{time_period}_{benchmark_name}_{datetime.now().isoformat()}"
    writer = SummaryWriter(log_dir=log_dir)

    # Convergence tracking
    convergence_data = {
        'iteration': [],
        'loss': [],
        'max_relative_error': [],
        'mean_relative_error': [],
        'gpu_memory_mb': [],
        'iteration_time_ms': []
    }
    
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"...starting optimization with up to {max_iterations} iterations")
    optimization_start_time = time.time()
    
    # Clear GPU cache if using GPU
    if use_gpu_actual:
        torch.cuda.empty_cache()

    for i in tqdm(range(max_iterations), desc=f"Optimizing {benchmark_name}"):
        iter_start_time = time.time()
        
        optimizer.zero_grad()
        new_weights = weights * (
            torch.clamp(
                weight_multiplier,
                min=weight_multiplier_min,
                max=weight_multiplier_max,
            )
        )
        outputs = (new_weights * output_matrix_tensor.T).sum(axis=1)
        weight_deviation = (
            (new_weights - original_weights).abs().sum()
            / original_weights.sum()
            * weight_deviation_penalty
            * original_loss_value
        )
        loss_value = (
            ((outputs + 1) / (target_array + 1) - 1) ** 2
        ).sum() + weight_deviation
        loss_value.backward()
        optimizer.step()
        
        iter_end_time = time.time()
        iteration_time_ms = (iter_end_time - iter_start_time) * 1000
        
        # Convergence tracking
        if i % logging_frequency == 0:
            # Calculate metrics
            rel_errors = ((outputs + 1) / (target_array + 1) - 1).abs()
            max_rel_error = rel_errors.max().item()
            mean_rel_error = rel_errors.mean().item()
            current_loss = loss_value.item()
            
            # GPU memory usage
            gpu_memory_mb = 0
            if use_gpu_actual:
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
            
            # Store convergence data
            convergence_data['iteration'].append(i)
            convergence_data['loss'].append(current_loss)
            convergence_data['max_relative_error'].append(max_rel_error)
            convergence_data['mean_relative_error'].append(mean_rel_error)
            convergence_data['gpu_memory_mb'].append(gpu_memory_mb)
            convergence_data['iteration_time_ms'].append(iteration_time_ms)
            
            # TensorBoard logging (reduced frequency to minimize GPU transfers)
            writer.add_scalar("Summary/Loss", current_loss, i)
            writer.add_scalar("Summary/Max_relative_error", max_rel_error, i)
            writer.add_scalar("Summary/Mean_relative_error", mean_rel_error, i)
            if use_gpu_actual:
                writer.add_scalar("Hardware/GPU_memory_MB", gpu_memory_mb, i)
            
            # Early stopping check
            if early_stopping:
                improvement_ratio = (best_loss - current_loss) / best_loss if best_loss != 0 else 0
                if improvement_ratio > early_stopping_threshold:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n...Early stopping at iteration {i} (no improvement for {early_stopping_patience} checks)")
                    break

    optimization_end_time = time.time()
    optimization_duration = optimization_end_time - optimization_start_time
    actual_iterations = len(convergence_data['iteration']) * logging_frequency if convergence_data['iteration'] else i + 1
    iterations_per_second = actual_iterations / optimization_duration

    print(f"...optimization completed in {optimization_duration:.1f} seconds")
    print(f"...optimization speed: {iterations_per_second:.1f} iterations/second")
    print(f"...final loss: {convergence_data['loss'][-1]:.6f}")
    print(f"...final max relative error: {convergence_data['max_relative_error'][-1]:.4f}")

    # Prepare results summary
    results_summary = {
        'benchmark_name': benchmark_name,
        'learning_rate': learning_rate,
        'max_iterations': max_iterations,
        'actual_iterations': actual_iterations,
        'early_stopped': early_stopping and patience_counter >= early_stopping_patience,
        'optimization_duration_seconds': optimization_duration,
        'iterations_per_second': iterations_per_second,
        'final_loss': convergence_data['loss'][-1] if convergence_data['loss'] else None,
        'final_max_relative_error': convergence_data['max_relative_error'][-1] if convergence_data['max_relative_error'] else None,
        'final_mean_relative_error': convergence_data['mean_relative_error'][-1] if convergence_data['mean_relative_error'] else None,
        'device': str(device),
        'gpu_name': gpu_name if use_gpu_actual else None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save convergence data
    if save_convergence_data:
        results_path = Path("claude/optimization-benchmarks")
        results_path.mkdir(exist_ok=True)
        
        # Save detailed convergence data
        convergence_df = pd.DataFrame(convergence_data)
        convergence_df.to_csv(results_path / f"convergence_{benchmark_name}.csv", index=False)
        
        # Save results summary
        with open(results_path / f"summary_{benchmark_name}.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"...convergence data saved to claude/optimization-benchmarks/")

    writer.close()

    # Move final weights back to CPU for numpy conversion
    final_weights = new_weights.detach().cpu().numpy()
    flat_file = flat_file.copy()
    flat_file["s006"] = final_weights
    
    return flat_file, results_summary


def run_optimization_benchmarks():
    """
    Run a series of optimization benchmarks with different parameters.
    """
    print("Starting optimization parameter benchmarks...")
    
    # Load or create test data
    from tmd.datasets.tmd import create_tmd_2021
    print("Creating TMD 2021 dataset...")
    tmd_data = create_tmd_2021()
    
    # Test different parameter combinations
    test_configs = [
        # Baseline (current implementation)
        {
            'name': 'baseline_lr_0.1',
            'learning_rate': 1e-1,
            'logging_frequency': 100,
            'early_stopping': False
        },
        # Different learning rates
        {
            'name': 'lower_lr_0.05',
            'learning_rate': 5e-2,
            'logging_frequency': 100,
            'early_stopping': False
        },
        {
            'name': 'lower_lr_0.01',
            'learning_rate': 1e-2,
            'logging_frequency': 100,
            'early_stopping': False
        },
        # Early stopping tests
        {
            'name': 'early_stopping_patient',
            'learning_rate': 1e-1,
            'logging_frequency': 50,
            'early_stopping': True,
            'early_stopping_patience': 50
        },
        {
            'name': 'early_stopping_aggressive',
            'learning_rate': 1e-1,
            'logging_frequency': 50,
            'early_stopping': True,
            'early_stopping_patience': 20
        },
        # Reduced logging frequency
        {
            'name': 'reduced_logging_500',
            'learning_rate': 1e-1,
            'logging_frequency': 500,
            'early_stopping': False
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*50}")
        print(f"Running benchmark: {config['name']}")
        print(f"{'='*50}")
        
        try:
            _, result_summary = enhanced_reweight(
                tmd_data.copy(),
                benchmark_name=config['name'],
                learning_rate=config.get('learning_rate', 1e-1),
                logging_frequency=config.get('logging_frequency', 100),
                early_stopping=config.get('early_stopping', False),
                early_stopping_patience=config.get('early_stopping_patience', 100),
                save_convergence_data=True
            )
            results.append(result_summary)
            
        except Exception as e:
            print(f"Error in benchmark {config['name']}: {e}")
            continue
    
    # Save combined results
    results_path = Path("claude/optimization-benchmarks")
    with open(results_path / "benchmark_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Benchmark Results Summary:")
    print(f"{'='*50}")
    
    for result in results:
        print(f"{result['benchmark_name']:25} | "
              f"Duration: {result['optimization_duration_seconds']:6.1f}s | "
              f"Speed: {result['iterations_per_second']:6.1f} it/s | "
              f"Final Loss: {result['final_loss']:8.6f}")
    
    return results


if __name__ == "__main__":
    run_optimization_benchmarks()