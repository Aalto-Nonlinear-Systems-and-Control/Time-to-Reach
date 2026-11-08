"""Write the paper's tables (arXiv:2411.05554) from results/summary.

    python -m scripts.paper_table [--columns krtb:setting_0 cora:setting_0 ...] [--out-dir results/summary]

Writes <out-dir>/paper_table.tex, the comparison table (label tab:benchmark), and
<out-dir>/paper_table_mas.tex, the initial sets and principal eigenvalues of the
multi-agent example (label tab: multi-agent initial sets), with the layout,
row order, benchmark names and abbreviations of the arXiv version. Both need the
paper's preamble (booktabs, multirow, adjustbox, threeparttable, colortbl, the
M column type and \\gc).

Cell labels: Rea / PRea (several reach-time windows for one target) / Unr /
Inc (KRTB bounds meet the horizon but no simulated witness) / Ukn (possible
intersection, no certificate) / N/A (the tool does not handle the benchmark itself;
a note gives the reason where the arXiv version has one) / S/E / OOM / T/O /
Err / $\\times$ (verdict contradicted by the simulation baseline) / -- (not run).
Times are the core computation times of scripts.aggregate, except KRTB's: its two
columns time the principal eigenpairs (eigenvalues and eigenfunction values on the
samples) and the reach-time bounds. The witness simulation behind Rea is timed in
results/summary/table.md and the report, not here.
"""

import argparse
import sys
from pathlib import Path

from scripts.aggregate import CORE_TIME
from scripts.evaluate_results import evaluate
from tools.common import jsonio, load_benchmark, paths

DEFAULT_COLUMNS = ["krtb:setting_0", "cora:setting_0", "juliareach:setting_0", "dreach:setting_0",
                   "hj:setting_0", "sosbc:setting_0"]
# (system label, dimension, [(benchmark ID, name in the paper)]) in the paper's order. No space after
# \\ in the labels: tex-fmt's table formatting drops the text before "\\ " in a cell.
SYSTEMS = [
    ("Duffing's\\\\oscillator\\\\(Example 1)", 2, [
        ("DUFF_FIN_REA_BOX", "DUFF_REA_BOX"),
        ("DUFF_FIN_REA_BALL", "DUFF_REA_CVX_LS"),
        ("DUFF_FIN_REA_NONCONVEX_LEVEL_SET", "DUFF_REA_NCVX_LS"),
        ("DUFF_FIN_REA_DISJOINT_BOX", "DUFF_REA_DSJT_BOX"),
        ("DUFF_FIN_UNR_BOX", "DUFF_UNR_BOX"),
        ("DUFF_FIN_UNR_BALL", "DUFF_UNR_CVX_LS")]),
    ("Stable \\\\limit cycle", 2, [
        ("NL_SLC_FIN_REA_BOX", "NL-SLC_REA_BOX"),
        ("NL_SLC_FIN_UNR_BOX", "NL-SLC_UNR_BOX")]),
    ("Saddle point\\\\system", 2, [
        ("NL_EIG_FIN_BACKWARD_REA_BOX", "NL-EIG_BWD_REA_BOX"),
        ("NL_EIG_FIN_BACKWARD_UNR_BOX", "NL-EIG_BWD_UNR_BOX")]),
    ("Cartpole\\\\(Example 2)", 4, [
        ("CP_LQR_FIN_REA_CONVERGE", "CP-LQR_REA_CVG"),
        ("CP_LQR_FIN_REA_UNSAFE", "CP-LQR_REA_USF"),
        ("CP_LQR_FIN_UNR_SAFE", "CP-LQR_UNR_SAF")]),
    ("Multi-agent\\\\consensus\\\\(Example 3)", 16, [
        ("MAS_CON_FIN_REA_BOX", "MAS-CON_REA_BOX"),
        ("MAS_CON_FIN_UNR_SAFE_0", "MAS-CON_UNR_SAF_0"),
        ("MAS_CON_FIN_UNR_SAFE_1", "MAS-CON_UNR_SAF_1")]),
]
LABEL = {
    "reachable_proven": "Rea", "reachable_witness": "Rea", "reachable_estimated": "Rea",
    "unreachable_proven": "Unr", "unreachable_estimated": "Unr",
    "reachable_possible": "Ukn", "unknown": "Ukn", "unsupported": "N/A",
    "set_explosion": "S/E", "out_of_memory": "OOM", "timeout": "T/O", "error": "Err",
}
NO_TIME = {"N/A", "S/E", "OOM", "T/O", "Err", "--"}
# Why a tool does not apply: (text in the record's detail, label, note or None). The SOS note is the one of
# the arXiv version, where the set-propagation tools have N/A without a note.
REASONS = [
    ("box-to-box", "N/A", None),
    ("non-polynomial", "N/A", "Not directly applicable due to trigonometric, exponential or fractional polynomial "
                              "functions."),
]
ABBREVIATIONS = [
    "Rea --- Reachable; PRea --- Periodically reachable; Unr --- Unreachable; Ukn --- Unknown; "
    "Inc --- Inconclusive.",
    "N/A --- Not applicable; S/E --- Set explosion; OOM --- Out of memory; "
    "T/O --- Time out; Err --- Error; "
    "$\\times$ --- Unsound result; -- --- Not run.",
    "For more details regarding the benchmark specifications, please follow the shared link for the code. "
    "\\protect\\footnotemark",
]


def _tex(name):
    return name.replace("_", "\\_")


def _time(t):
    return "$<$0.01" if t < 0.005 else f"{t:.2f}"


def _sum(record, keys):
    vals = [record["runtime_s"].get(k) for k in keys]
    return sum(vals) if all(v is not None for v in vals) else record["runtime_s"].get("total")


def tool_cell(tool, record, truth):
    """{'label', 'time', 'reason'} for one tool on one benchmark."""
    if record is None:
        return {"label": "--", "time": None, "reason": None}
    verdict = record["verdict"]
    label = LABEL[verdict]
    if tool == "dreach" and verdict == "reachable_possible":
        label = "$\\delta$-Rea"
    if evaluate(record, truth)["verdict_check"] == "contradicted":
        label = "$\\times$"
    reason = None
    if verdict == "unsupported":
        label, reason = next(((lab, note) for key, lab, note in REASONS if key in record["detail"]),
                             (label, record["detail"]))
    return {"label": label, "time": _sum(record, CORE_TIME[tool]), "reason": reason}


def krtb_cells(record, truth, num_targets):
    """The two KRTB cells: principal eigenpairs, and verdict with the reach-time bounds time."""
    if record is None:
        return [{"label": "--", "time": None, "reason": None}] * 2
    analytic = record["extra"]["eigenfunctions"]["method"] == "analytic"
    eig = {"label": "Analytical" if analytic else "", "time": None if analytic else
           record["runtime_s"]["eigenpairs"], "reason": None}
    cell = tool_cell("krtb", record, truth)
    if record["verdict"] == "unknown":
        cell["label"] = "Inc"
    if record["verdict"] == "reachable_witness" and len(record["reach_time_bounds"]) > num_targets:
        cell["label"] = "PRea"
    cell["time"] = record["runtime_s"]["bounds"]
    return [eig, cell]


def build_rows(columns):
    """[(system, dim, [(bid, paper name, [cell, ...])])]; KRTB yields two cells."""
    out = []
    for system, dim, benchmarks in SYSTEMS:
        rows = []
        for bid, name in benchmarks:
            truth = jsonio.read_json(paths.SUMMARY_DIR / "ground_truth" / bid / "simulation.json")
            cells = []
            for tool, setting in columns:
                path = paths.SUMMARY_DIR / tool / bid / f"{setting}.json"
                record = jsonio.read_json(path) if path.exists() else None
                if tool == "krtb":
                    cells += krtb_cells(record, truth, len(load_benchmark(bid).raw["target_sets"]))
                else:
                    cells.append(tool_cell(tool, record, truth))
            rows.append((bid, name, cells))
        out.append((system, dim, rows))
    return out


def cell_text(cell, notes):
    """Default LaTeX for a cell; `notes` maps reason -> footnote letter."""
    if cell["label"] == "Analytical":
        text = "\\scriptsize Analytical"
    elif cell["label"] == "":
        text = _time(cell["time"])
    elif cell["label"] in NO_TIME or cell["time"] is None:
        text = cell["label"]
    else:
        text = f"{cell['label']} ({_time(cell['time'])})"
    if cell["reason"]:
        text += f" \\tnote{{{notes[cell['reason']]}}}"
    return text


def render(rows, decorate=None):
    """The table* environment; decorate(bid, column index, cell, default text) may restyle cells."""
    reasons = [c["reason"] for _, _, rs in rows for _, _, cells in rs for c in cells if c["reason"]]
    notes = {r: chr(ord("a") + i) for i, r in enumerate(dict.fromkeys(reasons))}
    tn = {k: chr(ord("a") + len(notes) + i) for i, k in enumerate(("abbr", "abbr2", "link"))}
    lines = [
        "% Generated by `python -m scripts.paper_table` from results/summary; do not edit by hand.",
        "\\begin{table*}",
        "    [!htbp]",
        "    \\centering",
        "    \\aboverulesep = 0pt \\belowrulesep = 0pt",
        "    \\begin{threeparttable}",
        "    \\begin{adjustbox}",
        "        {width=\\textwidth,center} \\def\\arraystretch{1.4}",
        "        \\begin{tabular}{ |M{0.08\\textwidth} |M{0.03\\textwidth} | M{0.2\\textwidth} | M{0.08\\textwidth}"
        "  M{0.08\\textwidth} | M{0.08\\textwidth} M{0.11\\textwidth} |M{0.11\\textwidth}| M{0.10\\textwidth}"
        " | M{0.12\\textwidth}|}",
        "            \\toprule",
        "            \\multicolumn{3}{|c|}{\\textbf{Benchmark Problem}} & \\multicolumn{7}{c|}{\\textbf{Verification"
        " Result and Computation Time (s)}} \\\\",
        "            \\cmidrule{1-6} \\cmidrule{7-10}",
        "            \\midrule",
        f"            \\multirow{{2}}{{*}}{{\\textbf{{System}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Dim.}}}} & "
        f"\\multirow{{2}}{{*}}{{\\textbf{{Benchmark ID}}\\tnote{{{tn['link']}}}}} & \\multicolumn{{2}}{{c|}}"
        "{\\textbf{Ours}} & \\multicolumn{2}{c|}{\\textbf{Set-Propagation}} & \\textbf{Counterexample Searching}"
        " & \\textbf{Hamilton-Jacobi PDE} & \\textbf{Barrier Certificate \\cite{prajna2004safety}} \\\\",
        "            \\cmidrule{4-10}",
        "            & & & \\textbf{Principal eigenpairs} & \\textbf{Reach-time bounds} & \\textbf{CORA}"
        " \\cite{althoff2016cora} & \\textbf{JuliaReach} \\cite{bogomolov2019juliareach} & \\textbf{dReach}"
        " \\cite{kong2015dreach} & \\textbf{hj\\_reachability} & \\textbf{SOSTOOLS}"
        " \\cite{papachristodoulou2013sostools} \\\\",
        "            \\midrule",
    ]
    index = 0
    for g, (system, dim, benchmarks) in enumerate(rows):
        n = len(benchmarks)
        for i, (bid, name, cells) in enumerate(benchmarks):
            index += 1
            gc = "\\gc " if index % 2 else ""
            texts = []
            for j, c in enumerate(cells):
                default = cell_text(c, notes)
                texts.append(decorate(bid, j, c, default) if decorate else default)
            head = (f"\\multirow{{{n}}}{{*}}{{\\textbf{{\\shortstack{{{system}}}}}}} & \\multirow{{{n}}}{{*}}{{{dim}}}"
                    if i == 0 else "&")
            body = " & ".join(gc + t for t in [_tex(name)] + texts)
            lines.append(f"            {head} & {body} \\\\")
        if g < len(rows) - 1:
            lines.append("            \\midrule")
    lines += [
        "            \\bottomrule",
        "        \\end{tabular}",
        "    \\end{adjustbox}",
        "    \\begin{tablenotes}",
    ]
    lines += [f"        \\item[{k}] {r}" for r, k in notes.items()]
    lines += [f"        \\item[{tn[k]}] {t}" for k, t in zip(("abbr", "abbr2", "link"), ABBREVIATIONS)]
    lines += [
        "    \\end{tablenotes}",
        "    \\caption{Comparative evaluation of reachability analysis methodologies on benchmarks.}",
        "    \\label{tab:benchmark}",
        "    \\end{threeparttable}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def _eig(lam):
    re, im = lam
    return f"{re:.3f} \\pm {abs(im):.3f}j" if abs(im) > 1e-9 else f"{re:.3f}"


def render_mas(setting="setting_0"):
    """Initial sets and principal eigenvalues of the multi-agent example."""
    bm = load_benchmark("MAS_CON_FIN_REA_BOX")
    bounds = bm.raw["initial_sets"][0]["bounds"]
    record = jsonio.read_json(paths.SUMMARY_DIR / "krtb" / "MAS_CON_FIN_REA_BOX" / f"{setting}.json")
    lams = record["extra"]["eigenvalues"]
    # a complex pair is one entry
    entries, k = [], 0
    while k < len(lams):
        if abs(lams[k][1]) > 1e-9:
            entries.append(f"${_eig(lams[k])}$")
            k += 2
        else:
            entries.append(f"${lams[k][0]:.3f}$")
            k += 1
    rows = [entries[i:i + 2] for i in range(0, len(entries), 2)]
    lines = [
        "% Generated by `python -m scripts.paper_table` from results/summary; do not edit by hand.",
        "\\begin{table}[!hb]",
        "    \\centering",
        "    \\aboverulesep = 0pt \\belowrulesep = 0pt",
        "    \\begin{adjustbox}",
        "        {width=0.48\\textwidth,center,keepaspectratio}",
        "        \\begin{tabular}{|M{0.04\\textwidth}|M{0.12\\textwidth} M{0.12\\textwidth}|}",
        "            \\toprule \\multirow{2}{*}{Agent} & \\multicolumn{2}{c|}{$X_{0}^{i}$} \\\\",
        "            \\cmidrule{2-3} & $x_{i}(0)\\in$ & $y_{i}(0) \\in$ \\\\",
    ]
    for a in range(len(bounds) // 2):
        (x0, x1), (y0, y1) = bounds[2 * a], bounds[2 * a + 1]
        gray = "\\rowcolor[gray]{.90} " if a % 2 == 0 else ""
        rule = "\\midrule " if a == 0 else ""
        lines.append(f"            {rule}{gray}${a + 1}$ & $[{x0:.3f}, {x1:.3f}]$ & $[{y0:.3f}, {y1:.3f}]$ \\\\")
    lines += [
        "            \\bottomrule",
        "        \\end{tabular}",
        "        \\begin{tabular}{|M{0.11\\textwidth}|}",
        "            \\toprule Principal Eigenvalues \\\\",
    ]
    for i, pair in enumerate(rows):
        rule = "\\midrule " if i == 0 else ""
        lines.append(f"            {rule}{', '.join(pair)} \\\\")
    lines += [
        "            \\bottomrule",
        "        \\end{tabular}",
        "    \\end{adjustbox}",
        "    \\caption{Initial sets and principal eigenfunctions of the multi-agent consensus system.}",
        "    \\label{tab: multi-agent initial sets}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS,
                        help="tool:setting pairs in the paper's column order (KRTB first)")
    parser.add_argument("--out-dir", default=str(paths.SUMMARY_DIR))
    args = parser.parse_args(argv)
    columns = [tuple(c.split(":")) for c in args.columns]
    krtb_setting = dict(columns).get("krtb", "setting_0")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper_table.tex").write_text(render(build_rows(columns)))
    (out / "paper_table_mas.tex").write_text(render_mas(krtb_setting))
    print(f"wrote {out / 'paper_table.tex'} and {out / 'paper_table_mas.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
