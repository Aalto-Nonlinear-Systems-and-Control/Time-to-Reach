"""Build the comparison table from results/summary.

    python -m scripts.aggregate [--columns krtb:setting_0 hj:setting_0 ...] [--out results/summary/table]

Each cell shows the verdict class and the tool's core computation time; cells
whose verdict is contradicted by the simulation baseline are marked, as are
reach-time bounds that miss simulated in-target times. Writes <out>.md,
<out>.csv and <out>.tex and prints the Markdown table.
"""

import argparse
import csv
import sys
from pathlib import Path

from scripts.evaluate_results import evaluate
from tools.common import jsonio, list_benchmarks, paths

DEFAULT_COLUMNS = ["krtb:setting_0", "hj:setting_0", "juliareach:setting_0", "cora:setting_0",
                   "sosbc:setting_0", "dreach:setting_0"]
NAMES = {"krtb": "KRTB", "hj": "HJ", "juliareach": "JuliaReach", "cora": "CORA", "sosbc": "SOSBC",
         "dreach": "dReach"}
# Runtime phases that make up each tool's core computation: everything needed for
# the verdict, without one-off costs (JIT compilation, MATLAB start-up, CORA
# derivative generation, container start-up, input sampling).
CORE_TIME = {
    "krtb": ("krtb", "witness_search"),  # reach-time bounds + simulated witness
    "hj": ("solve",),  # includes the min over the initial set at every step
    "juliareach": ("solve", "check"),
    "cora": ("reach", "check"),
    "sosbc": ("setup", "solve"),  # SOS program construction + SDP solve
    "dreach": ("dreach",),
}
# Steps of the core time shown separately: (label, runtime key)
STEPS = {
    "krtb": (("eigenpairs", "eigenpairs"), ("bounds", "bounds"), ("witness", "witness_search")),
}
SYMBOL = {
    "unreachable_proven": "UNR proven", "reachable_proven": "REA proven", "reachable_witness": "REA witness",
    "reachable_possible": "REA possible", "reachable_estimated": "REA est.", "unreachable_estimated": "UNR est.",
    "unknown": "unknown", "unsupported": "n/a", "set_explosion": "set explosion", "out_of_memory": "out of memory",
    "timeout": "timeout", "error": "error",
}


def steps(tool, record):
    """[(label, seconds)] of the separately shown steps, or [] for tools without them."""
    return [(label, record["runtime_s"][key]) for label, key in STEPS.get(tool, ())]


def cell(tool, record, row):
    if record is None:
        return {"text": "-", "verdict": None, "time": None, "steps": []}
    phases = [record["runtime_s"].get(k) for k in CORE_TIME[tool]]
    t = sum(phases) if all(v is not None for v in phases) else record["runtime_s"].get("total")
    text = SYMBOL[record["verdict"]]
    split = steps(tool, record)
    if split:
        text += " (" + " + ".join(f"{label} {v:.3g} s" for label, v in split) + ")"
    elif record["verdict"] not in ("unsupported", "out_of_memory") and t is not None:
        text += f" ({t:.3g} s)"
    if row["verdict_check"] == "contradicted":
        text += " [CONTRADICTED]"
    if row.get("bounds_cover_simulation") is False:
        text += " [bounds miss]"
    return {"text": text, "verdict": record["verdict"], "time": t, "steps": split}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS, help="tool:setting pairs")
    parser.add_argument("--out", default=str(paths.SUMMARY_DIR / "table"))
    args = parser.parse_args(argv)
    columns = [tuple(c.split(":")) for c in args.columns]

    rows, platforms = [], set()
    for bid in list_benchmarks():
        truth = jsonio.read_json(paths.SUMMARY_DIR / "ground_truth" / bid / "simulation.json")
        cells = []
        for tool, setting in columns:
            path = paths.SUMMARY_DIR / tool / bid / f"{setting}.json"
            record = jsonio.read_json(path) if path.exists() else None
            if record is not None:
                platforms.add(record["provenance"]["platform"])
            cells.append(cell(tool, record, evaluate(record, truth) if record else {}))
        rows.append((bid, truth["expected_result"], cells))

    header = ["benchmark", "expected"] + [f"{NAMES[t]} ({s})" for t, s in columns]
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for bid, expected, cells in rows:
        md.append("| " + " | ".join([bid, expected] + [c["text"] for c in cells]) + " |")

    summary = ["", "| tool | " + " | ".join(SYMBOL.values()) + " |", "|" + "---|" * (len(SYMBOL) + 1)]
    for i, (tool, setting) in enumerate(columns):
        counts = [sum(1 for _, _, cells in rows if cells[i]["verdict"] == v) for v in SYMBOL]
        summary.append(f"| {NAMES[tool]} ({setting}) | " + " | ".join(str(c) for c in counts) + " |")
    note = [
        "",
        "Times are each tool's core computation in seconds: everything needed for the verdict, excluding",
        "JIT compilation, MATLAB start-up, CORA derivative generation and container start-up. KRTB's is",
        "split into the principal eigenpairs (eigenvalues and eigenfunction values on the samples), the",
        "reach-time bounds and the witness simulation. Measured on: " + "; ".join(sorted(platforms)) + ".",
    ]
    text = "\n".join(md + summary + note) + "\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(text)
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        keys = {t: ["verdict", "time_s"] + [f"{label}_s" for label, _ in STEPS.get(t, ())] for t, _ in columns}
        w.writerow(["benchmark", "expected"] + [f"{t}:{s}:{k}" for t, s in columns for k in keys[t]])
        for bid, expected, cells in rows:
            values = []
            for (t, _), c in zip(columns, cells):
                split = [v for _, v in c["steps"]] or [""] * (len(keys[t]) - 2)
                values += [c["verdict"] or "", c["time"] or ""] + split
            w.writerow([bid, expected] + values)
    tex = ["\\begin{tabular}{ll" + "l" * len(columns) + "}", "\\toprule",
           " & ".join(["Benchmark", "Expected"] + [NAMES[t] for t, _ in columns]) + " \\\\", "\\midrule"]
    for bid, expected, cells in rows:
        tex.append(" & ".join([bid.replace("_", "\\_"), expected] + [c["text"] for c in cells]) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    out.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
