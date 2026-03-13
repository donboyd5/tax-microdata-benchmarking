"""
End-to-end area weight pipeline: SOI data → targets → weights.

Four stages:
  1. prepare  — build enhanced targets from SOI + TMD data
  2. write    — write per-area target CSV files (base safe recipe)
  3. extend   — append SOI-shared and Census-shared targets (145 total)
  4. solve    — run Clarabel optimizer for each area (parallel)

Usage:
    # Full pipeline for all states (145 targets, 8 workers):
    python -m tmd.areas.prepare_and_solve --scope states --workers 8

    # Full pipeline for all CDs:
    python -m tmd.areas.prepare_and_solve --scope cds --workers 8

    # Full pipeline for everything:
    python -m tmd.areas.prepare_and_solve --scope all --workers 8

    # Only prepare + write targets (skip solving):
    python -m tmd.areas.prepare_and_solve --scope states --stage targets

    # Only solve (targets already written):
    python -m tmd.areas.prepare_and_solve \
        --scope states --stage solve --workers 8

    # Use 2022 SOI shares instead of 2021:
    python -m tmd.areas.prepare_and_solve --scope states --year 2022 --workers 8

    # Specific areas:
    python -m tmd.areas.prepare_and_solve --scope MN,MN01,MN02 --workers 4
"""

import argparse
import time
from pathlib import Path

from tmd.areas import AREAS_FOLDER

# --- Default paths ---------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_RECIPES = (
    _REPO_ROOT / "tmd" / "areas" / "targets" / "prepare" / "target_recipes"
)
_STATE_RECIPE = _RECIPES / "states_safe.json"
_CD_RECIPE = _RECIPES / "cds_final.json"
_STATE_VARMAP = _RECIPES / "state_variable_mapping_safe.csv"
_CD_VARMAP = _RECIPES / "cd_variable_mapping_allshares.csv"
_TARGET_DIR = AREAS_FOLDER / "targets"


def _prepare_and_write(
    scope,
    area_data_year=2021,
    national_data_year=0,
    pop_year=0,
):
    """
    Build enhanced targets and write per-area CSV files.

    Parameters
    ----------
    scope : str
        'states', 'cds', 'all', or comma-separated area codes.
    area_data_year : int
        SOI data year for geographic distribution.
    national_data_year : int
        TMD data year for national levels (0 = same as area_data_year).
    pop_year : int
        Population year (0 = same as area_data_year).

    Returns
    -------
    dict
        Mapping of area code → target count for each scope processed.
    """
    from tmd.areas.prepare.constants import AreaType
    from tmd.areas.prepare.target_file_writer import (
        write_area_target_files,
    )
    from tmd.areas.prepare.target_sharing import prepare_area_targets

    do_states, do_cds, specific = _parse_scope(scope)
    all_results = {}

    if do_states:
        print("Preparing state targets...")
        t0 = time.time()
        enhanced = prepare_area_targets(
            area_type=AreaType.STATE,
            area_data_year=area_data_year,
            national_data_year=national_data_year,
            pop_year=pop_year,
        )
        # Filter to specific areas if requested
        state_areas = None
        if specific:
            state_codes = [
                a.upper()
                for a in specific
                if len(a) == 2 and a.upper() != "XX"
            ]
            if state_codes:
                enhanced = enhanced[enhanced["area"].isin(state_codes)]
                state_areas = state_codes
        # Write base targets (safe recipe)
        result = write_area_target_files(
            recipe_path=_STATE_RECIPE,
            enhanced_targets=enhanced,
            variable_mapping_path=_STATE_VARMAP,
            output_dir=_TARGET_DIR,
        )
        base_count = next(iter(result.values()), 0)
        # Append extended targets (SOI-shared + Census-shared)
        from tmd.areas.prepare.extended_targets import (
            append_extended_targets,
        )

        ext_result = append_extended_targets(
            target_dir=_TARGET_DIR,
            enhanced_targets=enhanced,
            soi_year=area_data_year,
            areas=state_areas,
        )
        elapsed = time.time() - t0
        n_areas = len(ext_result) if ext_result else len(result)
        ext_count = next(iter(ext_result.values()), 0) if ext_result else 0
        print(
            f"  Wrote {n_areas} state target files: "
            f"{base_count} base + {ext_count - base_count} extended "
            f"= {ext_count} total ({elapsed:.1f}s)"
        )
        all_results.update(ext_result if ext_result else result)

    if do_cds:
        print("Preparing CD targets (with 117→118 crosswalk)...")
        t0 = time.time()
        enhanced = prepare_area_targets(
            area_type=AreaType.CD,
            area_data_year=area_data_year,
            national_data_year=national_data_year,
            pop_year=pop_year,
            apply_cd_crosswalk=True,
        )
        if specific:
            cd_codes = [
                a.upper() for a in specific if len(a) > 2 and a.upper() != "XX"
            ]
            if cd_codes:
                enhanced = enhanced[enhanced["area"].isin(cd_codes)]
        result = write_area_target_files(
            recipe_path=_CD_RECIPE,
            enhanced_targets=enhanced,
            variable_mapping_path=_CD_VARMAP,
            output_dir=_TARGET_DIR,
        )
        elapsed = time.time() - t0
        n_areas = len(result)
        print(f"  Wrote {n_areas} CD target files " f"({elapsed:.1f}s)")
        all_results.update(result)

    return all_results


def _solve(scope, num_workers=1, force=True):
    """
    Run the Clarabel solver for the specified areas.

    Parameters
    ----------
    scope : str
        'states', 'cds', 'all', or comma-separated area codes.
    num_workers : int
        Number of parallel worker processes.
    force : bool
        Recompute all areas even if up-to-date.
    """
    from tmd.areas.batch_weights import run_batch

    _, _, specific = _parse_scope(scope)
    if specific:
        area_filter = ",".join(a.lower() for a in specific)
    elif scope == "states":
        area_filter = "states"
    elif scope == "cds":
        area_filter = "cds"
    else:
        area_filter = "all"

    run_batch(
        num_workers=num_workers,
        area_filter=area_filter,
        force=force,
    )


def _parse_scope(scope):
    """
    Parse scope string into (do_states, do_cds, specific_list).

    Returns
    -------
    tuple
        (do_states: bool, do_cds: bool, specific: list or None)
    """
    scope_lower = scope.lower().strip()
    if scope_lower == "states":
        return True, False, None
    if scope_lower == "cds":
        return False, True, None
    if scope_lower == "all":
        return True, True, None
    # Comma-separated specific areas
    codes = [c.strip().upper() for c in scope.split(",") if c.strip()]
    has_states = any(len(c) == 2 for c in codes)
    has_cds = any(len(c) > 2 for c in codes)
    return has_states, has_cds, codes


def run_pipeline(
    scope="all",
    stage="all",
    num_workers=1,
    area_data_year=2021,
    national_data_year=0,
    pop_year=0,
    force=True,
):
    """
    Run the end-to-end area weight pipeline.

    Parameters
    ----------
    scope : str
        'states', 'cds', 'all', or comma-separated area codes.
    stage : str
        'all' (default), 'targets' (prepare+write only), or
        'solve' (solver only, targets must exist).
    num_workers : int
        Number of parallel solver workers.
    area_data_year : int
        SOI data year.
    national_data_year : int
        TMD national data year (0 = same as area_data_year).
    pop_year : int
        Population year (0 = same as area_data_year).
    force : bool
        Recompute solver even if weights are up-to-date.
    """
    t_total = time.time()

    if stage in ("all", "targets"):
        _prepare_and_write(
            scope=scope,
            area_data_year=area_data_year,
            national_data_year=national_data_year,
            pop_year=pop_year,
        )

    if stage in ("all", "solve"):
        _solve(
            scope=scope,
            num_workers=num_workers,
            force=force,
        )

    total = time.time() - t_total
    print(f"\nTotal pipeline time: {total:.1f}s")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="End-to-end area weight pipeline",
    )
    parser.add_argument(
        "--scope",
        default="all",
        help=(
            "'states', 'cds', 'all', or comma-separated area "
            "codes (e.g., 'MN,MN01,MN02')"
        ),
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "targets", "solve"],
        help=(
            "'all' = full pipeline, 'targets' = prepare+write "
            "only, 'solve' = solver only"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel solver workers",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2021,
        help="SOI area data year (default: 2021)",
    )
    parser.add_argument(
        "--national-year",
        type=int,
        default=0,
        help="TMD national data year (default: same as --year)",
    )
    parser.add_argument(
        "--pop-year",
        type=int,
        default=0,
        help="Population year (default: same as --year)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all areas even if up-to-date",
    )
    args = parser.parse_args()

    run_pipeline(
        scope=args.scope,
        stage=args.stage,
        num_workers=args.workers,
        area_data_year=args.year,
        national_data_year=args.national_year,
        pop_year=args.pop_year,
        force=args.force,
    )


if __name__ == "__main__":
    main()
