"""
Run a single reweight (GPU or CPU) and save weights.
Called by compare_gpu_cpu.py in a subprocess for PyTorch isolation.

Uses compare/pre_reweight_snapshot.csv.gz (saved by tmd.py during
make tmd_files) so the optimizer starts from the correct
pre-reweight weights.
"""

import argparse
import os
import sys

# Unbuffered output so log files are readable during execution
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import pandas as pd

sys.path.insert(0, ".")
from tmd.utils.reweight import reweight

SNAPSHOT_PATH = "compare/pre_reweight_snapshot.csv.gz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--grad-norm-tol", type=float, required=True)
    parser.add_argument("--penalty", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--min-mult", type=float, default=None)
    parser.add_argument("--max-mult", type=float, default=None)
    parser.add_argument("--use-lbfgsb", action="store_true")
    parser.add_argument("--use-scipy", action="store_true")
    parser.add_argument("--use-osqp", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(SNAPSHOT_PATH):
        print(
            f"ERROR: {SNAPSHOT_PATH} not found.\n"
            f"Run 'make clean && make tmd_files' first to generate it."
        )
        sys.exit(1)

    df = pd.read_csv(SNAPSHOT_PATH)
    print(f"Loaded {len(df)} records from {SNAPSHOT_PATH}")
    kwargs = {
        "use_gpu": args.use_gpu,
        "grad_norm_tol": args.grad_norm_tol,
    }
    if args.penalty is not None:
        kwargs["weight_deviation_penalty"] = args.penalty
    if args.min_mult is not None:
        kwargs["weight_multiplier_min"] = args.min_mult
    if args.max_mult is not None:
        kwargs["weight_multiplier_max"] = args.max_mult
    if args.use_lbfgsb:
        kwargs["use_lbfgsb"] = True
    if args.use_scipy:
        kwargs["use_scipy"] = True
    if args.use_osqp:
        kwargs["use_osqp"] = True
    df = reweight(df, 2021, **kwargs)
    df[["RECID", "s006"]].to_csv(args.output, index=False)
    print(f"Weights saved to {args.output}")


if __name__ == "__main__":
    main()
