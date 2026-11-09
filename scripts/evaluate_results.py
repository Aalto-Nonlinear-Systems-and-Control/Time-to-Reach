"""Judge every tool result against the simulation baseline.

For each results/summary/<tool>/<ID>/<setting>.json (tool != ground_truth) this
compares with results/summary/ground_truth/<ID>/simulation.json:

  verdict check   'consistent'    the tool's answer agrees with simulation,
                  'contradicted'  the tool claims unreachable but a simulated
                                  trajectory reaches the target (a definite error),
                  'unconfirmed'   the tool claims reachable but no simulated
                                  trajectory does,
                  'no claim'      unknown, unsupported, timeout, error, ...
  bounds check    for tools that report reach-time bounds: whether they cover the
                  simulated in-target time set (a sampled under-approximation of
                  the true set, so missing parts are definite misses).

Writes results/summary/evaluation/<tool>/<setting>.json and prints a table.
Exits non-zero if any verdict is contradicted by simulation.

    python -m scripts.evaluate_results [--tool krtb] [--setting setting_0]
"""

import argparse
import sys
from pathlib import Path

from tools.common import jsonio, paths
from tools.common.jsonio import from_jsonable_float
from tools.common.results import asserted_answer

TOL = 1e-6


def _intervals(raw):
    return [(from_jsonable_float(lo), from_jsonable_float(hi)) for lo, hi in raw or []]


def uncovered(sim, bounds):
    """Parts of the intervals in `sim` that no interval in `bounds` covers."""
    out = []
    for lo, hi in sim:
        pieces = [(lo, hi)]
        for blo, bhi in bounds:
            nxt = []
            for plo, phi in pieces:
                if bhi < plo or blo > phi:
                    nxt.append((plo, phi))
                    continue
                if plo < blo:
                    nxt.append((plo, blo))
                if bhi < phi:
                    nxt.append((bhi, phi))
            pieces = nxt
        out += [(a, b) for a, b in pieces if b - a > TOL or (a == lo and b == hi)]
    return out


def evaluate(record, truth):
    sim_reachable = truth["verdict"] == "reachable_witness"
    claim = asserted_answer(record["verdict"])
    if claim is None:
        verdict_check = "no claim"
    elif claim == "reachable":
        verdict_check = "consistent" if sim_reachable else "unconfirmed"
    else:
        verdict_check = "contradicted" if sim_reachable else "consistent"

    row = {
        "benchmark": record["benchmark"],
        "verdict": record["verdict"],
        "expected_result": record["expected_result"],
        "simulation": truth["verdict"],
        "verdict_check": verdict_check,
        "runtime_s": record["runtime_s"].get("total"),
    }
    bounds = record.get("reach_time_bounds")
    sim = _intervals(truth.get("extra", {}).get("in_target_times"))
    if bounds is not None and sim:
        missed = uncovered(sim, _intervals(bounds))
        row["bounds"] = bounds
        row["simulated_in_target_times"] = sim
        row["bounds_cover_simulation"] = not missed
        row["missed_times"] = missed
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tool", help="only this tool")
    parser.add_argument("--setting", help="only this setting")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    groups = {}
    for path in sorted(paths.SUMMARY_DIR.glob("*/*/*.json")):
        tool, bid, setting = path.parts[-3], path.parts[-2], Path(path.parts[-1]).stem
        if tool in ("ground_truth", "evaluation"):
            continue
        if (args.tool and tool != args.tool) or (args.setting and setting != args.setting):
            continue
        truth_path = paths.SUMMARY_DIR / "ground_truth" / bid / "simulation.json"
        if not truth_path.exists():
            print(f"missing simulation baseline for {bid}; run python -m scripts.check_benchmarks")
            return 1
        groups.setdefault((tool, setting), []).append(evaluate(jsonio.read_json(path), jsonio.read_json(truth_path)))

    contradicted = 0
    for (tool, setting), rows in sorted(groups.items()):
        print(f"\n## {tool} / {setting}\n")
        print("| benchmark | expected | verdict | vs. simulation | bounds cover simulated times |")
        print("|---|---|---|---|---|")
        for r in rows:
            cover = "-"
            if "bounds_cover_simulation" in r:
                cover = "yes" if r["bounds_cover_simulation"] else "no: misses " + ", ".join(
                    f"[{a:.4g}, {b:.4g}]" for a, b in r["missed_times"])
            print(f"| {r['benchmark']} | {r['expected_result']} | {r['verdict']} | {r['verdict_check']} | {cover} |")
        contradicted += sum(r["verdict_check"] == "contradicted" for r in rows)
        if not args.no_write:
            jsonio.write_json(paths.SUMMARY_DIR / "evaluation" / tool / f"{setting}.json",
                              {"tool": tool, "setting": setting, "rows": rows})
    return 1 if contradicted else 0


if __name__ == "__main__":
    sys.exit(main())
