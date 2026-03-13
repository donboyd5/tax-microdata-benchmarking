"""
Cross-state quality summary report.

Parses solver logs for all states and produces a summary showing:
  - Solve status and timing
  - Target accuracy (hit rate, mean/max error)
  - Weight distortion (RMSE, percentiles)
  - Violated targets by variable

Usage:
    python -m tmd.areas.quality_report
    python -m tmd.areas.quality_report --scope CA,WY
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from tmd.areas import AREAS_FOLDER
from tmd.areas.prepare.constants import ALL_STATES

WEIGHT_DIR = AREAS_FOLDER / "weights"

# Decode raw constraint descriptions into human-readable labels
_CNT_LABELS = {0: "amt", 1: "returns", 2: "nz-count"}
_FS_LABELS = {0: "all", 1: "single", 2: "MFJ", 4: "HoH"}


def _humanize_desc(desc: str) -> str:
    """
    Turn 'c00100/cnt=1/scope=1/agi=[500000.0,1000000.0)/fs=4'
    into 'c00100 returns HoH $500K-$1M'.
    """
    parts = desc.split("/")
    varname = parts[0]
    attrs = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            attrs[k] = v

    cnt = int(attrs.get("cnt", -1))
    fs = int(attrs.get("fs", 0))
    agi_raw = attrs.get("agi", "")

    cnt_label = _CNT_LABELS.get(cnt, f"cnt{cnt}")
    fs_label = _FS_LABELS.get(fs, f"fs{fs}")

    # Parse AGI range like [500000.0,1000000.0)
    agi_label = ""
    m = re.match(r"\[([^,]+),([^)]+)\)", agi_raw)
    if m:
        lo_s, hi_s = m.group(1), m.group(2)
        lo = float(lo_s)
        hi = float(hi_s)
        if lo < -1e10:
            agi_label = f"<${hi / 1000:.0f}K"
        elif hi > 1e10:
            agi_label = f"${lo / 1000:.0f}K+"
        else:
            agi_label = f"${lo / 1000:.0f}K-${hi / 1000:.0f}K"

    pieces = [varname, cnt_label]
    if fs != 0:
        pieces.append(fs_label)
    if agi_label:
        pieces.append(agi_label)
    return " ".join(pieces)


def parse_log(logpath: Path) -> dict:
    """Parse a single area solver log file into a summary dict."""
    if not logpath.exists():
        return {"status": "NO LOG"}
    log = logpath.read_text()

    result = {"status": "UNKNOWN"}

    # Solve status
    m = re.search(r"Solver status: (\S+)", log)
    if m:
        result["status"] = m.group(1)
    if "PrimalInfeasible" in log or "FAILED" in log:
        result["status"] = "FAILED"

    # Solve time
    m = re.search(r"Solve time: ([\d.]+)s", log)
    if m:
        result["solve_time"] = float(m.group(1))

    # Target accuracy
    m = re.search(r"mean \|relative error\|: ([\d.]+)", log)
    if m:
        result["mean_err"] = float(m.group(1))
    m = re.search(r"max  \|relative error\|: ([\d.]+)", log)
    if m:
        result["max_err"] = float(m.group(1))
    m = re.search(r"targets hit: (\d+)/(\d+)", log)
    if m:
        result["targets_hit"] = int(m.group(1))
        result["targets_total"] = int(m.group(2))
    m_viol = re.search(r"VIOLATED: (\d+) targets", log)
    result["n_violated"] = int(m_viol.group(1)) if m_viol else 0

    # Weight distortion
    m = re.search(
        r"min=([\d.]+), p5=([\d.]+), median=([\d.]+), "
        r"p95=([\d.]+), max=([\d.]+)",
        log,
    )
    if m:
        result["w_min"] = float(m.group(1))
        result["w_p5"] = float(m.group(2))
        result["w_median"] = float(m.group(3))
        result["w_p95"] = float(m.group(4))
        result["w_max"] = float(m.group(5))
    m = re.search(r"RMSE from 1.0: ([\d.]+)", log)
    if m:
        result["w_rmse"] = float(m.group(1))

    # Violated target details — lines after "VIOLATED: N targets"
    # Each line: "  0.400% | target=  29 | achieved=  29 | varname/cnt=.../..."
    violated = []
    in_violated = False
    for line in log.splitlines():
        if "VIOLATED:" in line and "targets" in line:
            in_violated = True
            continue
        if in_violated:
            m_det = re.match(
                r"\s+([\d.]+)%\s*\|\s*target=\s*([\d.]+)\s*\|"
                r"\s*achieved=\s*([\d.]+)\s*\|"
                r"\s*(\S+/cnt=\d+/scope=\d+/agi=.*?/fs=\d+)",
                line,
            )
            if m_det:
                violated.append(
                    {
                        "pct_err": float(m_det.group(1)),
                        "target": float(m_det.group(2)),
                        "achieved": float(m_det.group(3)),
                        "desc": m_det.group(4),
                    }
                )
            else:
                in_violated = False
    result["violated_details"] = violated

    return result


def generate_report(areas=None):
    """Generate cross-state quality summary report."""
    if areas is None:
        areas = ALL_STATES

    rows = []
    for st in areas:
        logpath = WEIGHT_DIR / f"{st.lower()}.log"
        info = parse_log(logpath)
        info["state"] = st
        rows.append(info)

    df = pd.DataFrame(rows)

    # Summary statistics
    solved = df[df["status"].isin(["Solved", "AlmostSolved"])]
    failed = df[df["status"] == "FAILED"]
    n_states = len(df)
    n_solved = len(solved)
    n_failed = len(failed)
    n_violated_states = (solved["n_violated"] > 0).sum()
    total_violated = solved["n_violated"].sum()

    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-STATE QUALITY SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Overall
    lines.append(f"States: {n_states}")
    lines.append(f"Solved: {n_solved}")
    lines.append(f"Failed: {n_failed}")
    if n_failed > 0:
        lines.append(
            f"  Failed: {', '.join(failed['state'].tolist())}"
        )
    lines.append(
        f"States with violated targets: "
        f"{n_violated_states}/{n_solved}"
    )
    lines.append(f"Total violated targets: {int(total_violated)}")
    lines.append("")

    # Target accuracy
    if not solved.empty and "mean_err" in solved.columns:
        lines.append("TARGET ACCURACY:")
        lines.append(
            f"  Mean |relative error|:  "
            f"avg={solved['mean_err'].mean():.4f}, "
            f"max={solved['mean_err'].max():.4f}"
        )
        lines.append(
            f"  Max |relative error|:   "
            f"avg={solved['max_err'].mean():.4f}, "
            f"max={solved['max_err'].max():.4f}"
        )
        if "targets_hit" in solved.columns:
            total_t = solved["targets_total"].iloc[0]
            hit_pcts = solved["targets_hit"] / solved["targets_total"] * 100
            lines.append(
                f"  Hit rate:  "
                f"avg={hit_pcts.mean():.1f}%, "
                f"min={hit_pcts.min():.1f}% "
                f"(out of {total_t} targets)"
            )
        lines.append("")

    # Weight distortion
    if not solved.empty and "w_rmse" in solved.columns:
        lines.append("WEIGHT DISTORTION (multiplier from 1.0):")
        lines.append(
            f"  RMSE:   avg={solved['w_rmse'].mean():.3f}, "
            f"max={solved['w_rmse'].max():.3f}"
        )
        lines.append(
            f"  Median: avg={solved['w_median'].mean():.3f}, "
            f"range=[{solved['w_median'].min():.3f}, "
            f"{solved['w_median'].max():.3f}]"
        )
        lines.append(
            f"  P95:    avg={solved['w_p95'].mean():.3f}, "
            f"max={solved['w_p95'].max():.3f}"
        )
        lines.append(
            f"  Max:    avg={solved['w_max'].mean():.1f}, "
            f"max={solved['w_max'].max():.1f}"
        )
        lines.append("")

    # Per-state table
    lines.append("PER-STATE DETAIL:")
    lines.append(
        "  Err cols = |relative error| (fraction); "
        "weight cols = multiplier on national weight (1.0 = unchanged)"
    )
    header = (
        f"{'St':<4} {'Status':<14} {'Hit':>5} {'Tot':>5} "
        f"{'Viol':>5} {'MeanErr':>8} {'MaxErr':>8} "
        f"{'wRMSE':>7} {'wMed':>7} {'wP95':>7} {'wMax':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.iterrows():
        hit = int(row.get("targets_hit", 0))
        tot = int(row.get("targets_total", 0))
        viol = int(row.get("n_violated", 0))
        me = row.get("mean_err", 0)
        mx = row.get("max_err", 0)
        rmse = row.get("w_rmse", 0)
        med = row.get("w_median", 0)
        p95 = row.get("w_p95", 0)
        wmax = row.get("w_max", 0)
        lines.append(
            f"{row['state']:<4} {row['status']:<14} {hit:>5} {tot:>5} "
            f"{viol:>5} {me:>8.4f} {mx:>8.4f} "
            f"{rmse:>7.3f} {med:>7.3f} {p95:>7.3f} {wmax:>8.1f}"
        )
    lines.append("")

    # Violated targets by variable
    all_violated = []
    for _, row in df.iterrows():
        for v in row.get("violated_details", []):
            desc = v["desc"]
            varname = desc.split("/")[0]
            cnt_m = re.search(r"cnt=(\d+)", desc)
            cnt_type = int(cnt_m.group(1)) if cnt_m else -1
            abs_miss = abs(v["achieved"] - v["target"])
            all_violated.append(
                {
                    "state": row["state"],
                    "varname": varname,
                    "cnt_type": cnt_type,
                    "pct_err": v["pct_err"],
                    "target": v["target"],
                    "achieved": v["achieved"],
                    "abs_miss": abs_miss,
                    "desc": desc,
                }
            )
    if all_violated:
        vdf = pd.DataFrame(all_violated)
        var_counts = vdf["varname"].value_counts()
        lines.append("VIOLATIONS BY VARIABLE:")
        for var, cnt in var_counts.items():
            states_with = sorted(
                vdf[vdf["varname"] == var]["state"].unique()
            )
            lines.append(
                f"  {var}: {cnt} violations across "
                f"{len(states_with)} states"
            )
        lines.append("")

        # Worst states
        state_counts = vdf["state"].value_counts().head(10)
        lines.append("STATES WITH MOST VIOLATIONS:")
        for st, cnt in state_counts.items():
            lines.append(f"  {st}: {cnt} violated")
        lines.append("")

        # Worst 5 amount violations (cnt=0)
        amt_viol = vdf[vdf["cnt_type"] == 0].sort_values(
            ["pct_err", "abs_miss"], ascending=[False, False]
        )
        lines.append("WORST 5 AMOUNT TARGET VIOLATIONS:")
        if amt_viol.empty:
            lines.append("  (none — all amount targets met)")
        else:
            for _, r in amt_viol.head(5).iterrows():
                lines.append(
                    f"  {r['state']:<4} {r['pct_err']:.3f}% "
                    f"target=${r['target']:>15,.0f}  "
                    f"achieved=${r['achieved']:>15,.0f}  "
                    f"miss=${r['abs_miss']:>12,.0f}  "
                    f"{_humanize_desc(r['desc'])}"
                )
        lines.append("")

        # Worst 5 count violations (cnt=1 or cnt=2)
        cnt_viol = vdf[vdf["cnt_type"].isin([1, 2])].sort_values(
            ["pct_err", "abs_miss"], ascending=[False, False]
        )
        lines.append("WORST 5 COUNT TARGET VIOLATIONS:")
        if cnt_viol.empty:
            lines.append("  (none — all count targets met)")
        else:
            for _, r in cnt_viol.head(5).iterrows():
                lines.append(
                    f"  {r['state']:<4} {r['pct_err']:.3f}% "
                    f"target={r['target']:>12,.0f}  "
                    f"achieved={r['achieved']:>12,.0f}  "
                    f"miss={r['abs_miss']:>8,.0f}  "
                    f"{_humanize_desc(r['desc'])}"
                )
        lines.append("")

    report = "\n".join(lines)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Cross-state quality summary report",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help="Comma-separated state codes (default: all states)",
    )
    args = parser.parse_args()

    areas = None
    if args.scope:
        areas = [s.strip().upper() for s in args.scope.split(",")]

    report = generate_report(areas)
    print(report)


if __name__ == "__main__":
    main()
