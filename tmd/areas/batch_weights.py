"""
Batch area weight optimization — parallel processing for all areas.

Improvements over the original make_all.py:
  - TMD data loaded once per worker (not once per area).
  - Progress reporting with ETA.
  - Can generate target files from the Python pipeline before
    running the optimizer, or use existing target CSV files.
  - Uses concurrent.futures for clean parallel execution.

Usage:
    python -m tmd.areas.batch_weights --workers 8 --areas states
    python -m tmd.areas.batch_weights --workers 8 --areas cds
    python -m tmd.areas.batch_weights --workers 8 --areas all
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

from tmd.areas import AREAS_FOLDER
from tmd.areas.create_area_weights import valid_area

# Module-level cache for TMD data (one per worker process)
_WORKER_VDF = None
_WORKER_POP = None


def _init_worker():
    """Load TMD data once per worker process."""
    global _WORKER_VDF, _WORKER_POP
    if _WORKER_VDF is not None:
        return
    from tmd.areas.create_area_weights_clarabel import (
        INFILE_PATH,
        POPFILE_PATH,
        TAXCALC_AGI_CACHE,
    )

    _WORKER_VDF = pd.read_csv(INFILE_PATH)
    _WORKER_VDF["c00100"] = np.load(TAXCALC_AGI_CACHE)
    with open(POPFILE_PATH, "r", encoding="utf-8") as pf:
        _WORKER_POP = yaml.safe_load(pf.read())


def _solve_one_area(area):
    """
    Solve area weights for one area using cached worker data.

    Returns (area, elapsed_seconds, n_targets, n_violated, status).
    """
    _init_worker()
    from tmd.areas.create_area_weights_clarabel import (
        AREA_CONSTRAINT_TOL,
        AREA_MAX_ITER,
        AREA_MULTIPLIER_MAX,
        AREA_MULTIPLIER_MIN,
        AREA_SLACK_PENALTY,
        FIRST_YEAR,
        LAST_YEAR,
        _build_constraint_matrix,
        _drop_impossible_targets,
        _print_multiplier_diagnostics,
        _print_slack_diagnostics,
        _print_target_diagnostics,
        _read_params,
        _solve_area_qp,
    )
    import io

    t0 = time.time()
    out = io.StringIO()

    # Build and solve
    vdf = _WORKER_VDF
    params = _read_params(area, out)
    constraint_tol = params.get(
        "constraint_tol",
        params.get("target_ratio_tolerance", AREA_CONSTRAINT_TOL),
    )
    slack_penalty = params.get("slack_penalty", AREA_SLACK_PENALTY)
    max_iter = params.get("max_iter", AREA_MAX_ITER)
    multiplier_min = params.get("multiplier_min", AREA_MULTIPLIER_MIN)
    multiplier_max = params.get("multiplier_max", AREA_MULTIPLIER_MAX)

    B_csc, targets, labels, pop_share = _build_constraint_matrix(
        area, vdf, out
    )
    B_csc, targets, labels = _drop_impossible_targets(
        B_csc, targets, labels, out
    )

    n_records = B_csc.shape[1]
    x_opt, s_lo, s_hi, info = _solve_area_qp(
        B_csc,
        targets,
        labels,
        n_records,
        constraint_tol=constraint_tol,
        slack_penalty=slack_penalty,
        max_iter=max_iter,
        multiplier_min=multiplier_min,
        multiplier_max=multiplier_max,
        out=out,
    )

    # Diagnostics
    n_violated = _print_target_diagnostics(
        x_opt, B_csc, targets, labels, constraint_tol, out
    )
    _print_multiplier_diagnostics(x_opt, out)
    _print_slack_diagnostics(s_lo, s_hi, targets, labels, out)

    # Write log
    logpath = AREAS_FOLDER / "weights" / f"{area}.log"
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with open(logpath, "w", encoding="utf-8") as f:
        f.write(out.getvalue())

    # Write weights file
    w0 = pop_share * vdf.s006.values
    wght_area = x_opt * w0

    wdict = {f"WT{FIRST_YEAR}": wght_area}
    cum_pop_growth = 1.0
    pop = _WORKER_POP
    for year in range(FIRST_YEAR + 1, LAST_YEAR + 1):
        annual_pop_growth = pop[year] / pop[year - 1]
        cum_pop_growth *= annual_pop_growth
        wdict[f"WT{year}"] = wght_area * cum_pop_growth

    wdf = pd.DataFrame.from_dict(wdict)
    awpath = AREAS_FOLDER / "weights" / f"{area}_tmd_weights.csv.gz"
    wdf.to_csv(
        awpath,
        index=False,
        float_format="%.5f",
        compression="gzip",
    )

    elapsed = time.time() - t0
    return area, elapsed, len(targets), n_violated, info["status"]


def _list_target_areas():
    """Return sorted list of area codes with target files."""
    tfolder = AREAS_FOLDER / "targets"
    tpaths = sorted(tfolder.glob("*_targets.csv"))
    areas = []
    for tpath in tpaths:
        area = tpath.name.split("_")[0]
        if valid_area(area):
            areas.append(area)
    return areas


def _filter_areas(areas, area_filter):
    """Filter areas by type: 'states', 'cds', or 'all'."""
    if area_filter == "all":
        return areas
    if area_filter == "states":
        return [a for a in areas if len(a) == 2]
    if area_filter == "cds":
        return [a for a in areas if len(a) > 2]
    # Treat as comma-separated list
    requested = [a.strip() for a in area_filter.split(",")]
    return [a for a in areas if a in requested]


def run_batch(
    num_workers=1,
    area_filter="all",
    force=False,
):
    """
    Run area weight optimization for multiple areas in parallel.

    Parameters
    ----------
    num_workers : int
        Number of parallel worker processes.
    area_filter : str
        'states', 'cds', 'all', or comma-separated area codes.
    force : bool
        If True, recompute all areas even if up-to-date.
    """
    all_areas = _list_target_areas()
    areas = _filter_areas(all_areas, area_filter)

    if not areas:
        print("No areas to process.")
        return

    # Filter to out-of-date areas unless force=True
    if not force:
        from tmd.areas.make_all import time_of_newest_other_dependency

        newest_dep = time_of_newest_other_dependency()
        todo = []
        for area in areas:
            wpath = AREAS_FOLDER / "weights" / f"{area}_tmd_weights.csv.gz"
            tpath = AREAS_FOLDER / "targets" / f"{area}_targets.csv"
            if wpath.exists():
                wtime = wpath.stat().st_mtime
                ttime = tpath.stat().st_mtime
                if wtime > max(newest_dep, ttime):
                    continue
            todo.append(area)
        areas = todo

    if not areas:
        print("All areas up-to-date. Use --force to recompute.")
        return

    n = len(areas)
    print(f"Processing {n} areas with {num_workers} workers:")
    for i, area in enumerate(areas):
        sys.stdout.write(f"{area:>7s}")
        if (i + 1) % 10 == 0:
            sys.stdout.write("\n")
    if n % 10 != 0:
        sys.stdout.write("\n")
    print()

    # Ensure weights directory exists
    (AREAS_FOLDER / "weights").mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    completed = 0
    violated_areas = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
    ) as executor:
        futures = {
            executor.submit(_solve_one_area, area): area for area in areas
        }
        for future in as_completed(futures):
            area = futures[future]
            try:
                area_code, elapsed, n_tgt, n_viol, status = future.result()
                completed += 1
                elapsed_total = time.time() - t_start
                avg_per_area = elapsed_total / completed
                remaining = (
                    (n - completed) * avg_per_area / max(num_workers, 1)
                )

                viol_str = ""
                if n_viol > 0:
                    viol_str = f" [{n_viol} VIOLATED]"
                    violated_areas.append((area_code, n_viol))

                print(
                    f"  {area_code:>6s}: {elapsed:5.1f}s"
                    f" ({n_tgt} targets, {status})"
                    f"{viol_str}"
                    f"  [{completed}/{n},"
                    f" ~{remaining:.0f}s remaining]"
                )
            except Exception as exc:
                completed += 1
                print(f"  {area:>6s}: FAILED - {exc}")

    total = time.time() - t_start
    print(f"\nCompleted {completed}/{n} areas in {total:.1f}s")
    if violated_areas:
        print(f"Areas with violated targets: {len(violated_areas)}")
        for area_code, n_viol in sorted(violated_areas):
            print(f"  {area_code}: {n_viol} violated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch area weight optimization"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--areas",
        type=str,
        default="all",
        help="'states', 'cds', 'all', or comma-separated codes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all areas even if up-to-date",
    )
    args = parser.parse_args()
    run_batch(
        num_workers=args.workers,
        area_filter=args.areas,
        force=args.force,
    )
