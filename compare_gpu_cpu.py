"""
Compare GPU vs CPU reweighting at different grad_norm tolerances.

Runs each reweight in a separate subprocess for complete PyTorch
isolation (matching the production pipeline in tmd.py).

Usage:
    python compare_gpu_cpu.py                    # tries 1e-6, 1e-7, 1e-8
    python compare_gpu_cpu.py --tols 1e-6 1e-7   # specific tolerances
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def run_reweight(
    use_gpu, grad_norm_tol, output_path, log_path, penalty=None
):
    """Run a single reweight in a subprocess, saving weights and log."""
    cmd = [
        sys.executable,
        "run_single_reweight.py",
        f"--grad-norm-tol={grad_norm_tol}",
        f"--output={output_path}",
    ]
    if use_gpu:
        cmd.append("--use-gpu")
    if penalty is not None:
        cmd.append(f"--penalty={penalty}")

    device = "GPU" if use_gpu else "CPU"
    print(f"  Starting {device} run (grad_norm_tol={grad_norm_tol:.0e})...")

    with open(log_path, "w") as log:
        result = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT
        )

    if result.returncode != 0:
        print(f"  ERROR: {device} run failed (exit code {result.returncode})")
        print(f"  See log: {log_path}")
        return False
    print(f"  {device} run complete. Log: {log_path}")
    return True


def compare_weights(gpu_path, cpu_path, tol_label):
    """Compare GPU and CPU weights, print diagnostics."""
    gpu = pd.read_csv(gpu_path)
    cpu = pd.read_csv(cpu_path)

    gpu_w = gpu["s006"].values
    cpu_w = cpu["s006"].values

    diff = np.abs(gpu_w - cpu_w)
    rel_diff = diff / np.maximum(np.abs(gpu_w), 1e-10)

    print(f"\n{'='*60}")
    print(f"COMPARISON: grad_norm_tol = {tol_label}")
    print(f"{'='*60}")

    # np.allclose with default tolerances (rtol=1e-5, atol=1e-8)
    close = np.allclose(gpu_w, cpu_w)
    print(f"np.allclose(gpu, cpu) [defaults]: {close}")

    print(f"  max  |diff|:     {diff.max():.2e}")
    print(f"  mean |diff|:     {diff.mean():.2e}")
    print(f"  median |diff|:   {np.median(diff):.2e}")
    print(f"  max  |rel diff|: {rel_diff.max():.2e}")
    print(f"  mean |rel diff|: {rel_diff.mean():.2e}")

    # Check at multiple rtol levels
    for rtol in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        c = np.allclose(gpu_w, cpu_w, rtol=rtol, atol=1e-8)
        print(f"  np.allclose(rtol={rtol:.0e}): {c}")

    # Show worst 10 records
    worst_idx = np.argsort(rel_diff)[::-1][:10]
    print("  Worst 10 records (by |rel diff|):")
    for i in worst_idx:
        print(
            f"    RECID={int(gpu['RECID'].iloc[i]):>7d}  "
            f"gpu={gpu_w[i]:.10f}  cpu={cpu_w[i]:.10f}  "
            f"|diff|={diff[i]:.2e}  |rel|={rel_diff[i]:.2e}"
        )

    return close


def main():
    parser = argparse.ArgumentParser(
        description="Compare GPU vs CPU reweighting"
    )
    parser.add_argument(
        "--tols",
        type=float,
        nargs="+",
        default=[1e-6, 1e-7, 1e-8],
        help="Grad norm tolerances to try (default: 1e-6 1e-7 1e-8)",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="Override weight deviation penalty (default: use config)",
    )
    args = parser.parse_args()

    os.makedirs("compare", exist_ok=True)

    for tol in args.tols:
        tag = f"{tol:.0e}"
        gpu_weights = f"compare/gpu_weights_{tag}.csv"
        cpu_weights = f"compare/cpu_weights_{tag}.csv"
        gpu_log = f"compare/gpu_log_{tag}.txt"
        cpu_log = f"compare/cpu_log_{tag}.txt"

        print(f"\n{'#'*60}")
        print(f"# TOLERANCE: {tag}")
        print(f"{'#'*60}")

        ok = run_reweight(
            True, tol, gpu_weights, gpu_log, penalty=args.penalty
        )
        if not ok:
            continue

        ok = run_reweight(
            False, tol, cpu_weights, cpu_log, penalty=args.penalty
        )
        if not ok:
            continue

        close = compare_weights(gpu_weights, cpu_weights, tag)

        if close:
            print(f"\nSUCCESS: np.allclose() achieved at "
                  f"grad_norm_tol={tag}")
            print("No need to try tighter tolerances.")
            break
        else:
            print(f"\nFAILED: np.allclose() not achieved at "
                  f"grad_norm_tol={tag}")
            if tol != args.tols[-1]:
                print("Trying tighter tolerance...")
    else:
        print(f"\nNo tolerance in {args.tols} achieved np.allclose().")

    print("\nAll output files in compare/:")
    for f in sorted(os.listdir("compare")):
        size = os.path.getsize(f"compare/{f}")
        print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
