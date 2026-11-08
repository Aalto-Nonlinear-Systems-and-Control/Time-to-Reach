"""Run JuliaReach (ReachabilityAnalysis.jl, TMJets) on benchmarks.

    python -m tools.juliareach.run --benchmark DUFF_FIN_REA_BOX [--setting setting_0]
    python -m tools.juliareach.run --all

Calls tools/juliareach/run.jl in the pinned Julia project (Project.toml and
Manifest.toml next to it) and records the result. JuliaReach is run on what it
handles itself: a forward problem from one initial box to one target box
(tools.common.support); every other benchmark is unsupported. Verdicts:
  some flowpipe segment lies inside the target   -> reachable_proven
  horizon covered, every segment disjoint        -> unreachable_proven
  horizon covered, some segment may intersect    -> reachable_possible (no proof:
                                                     the segments are over-approximations)
  flowpipe stopped before the end of the horizon -> set_explosion (TMJets stops
                                                     at maxsteps or when a step
                                                     cannot be validated)
  no result within timeout_s                     -> timeout
The time spans of segments that may intersect the target are reported as
reach-time bounds (an over-approximation of the in-target times).
"""

import argparse
import shutil
import subprocess
import sys
import time

from tools.common import jsonio, load_benchmark, make_result, paths, write_result
from tools.common.settings import configured_benchmarks, load_settings
from tools.common.support import box_to_box_reason

TOOL = "juliareach"
PROJECT = paths.REPO_ROOT / "tools" / "juliareach"


def julia_command(version):
    exe = shutil.which("julia")
    if exe is None:
        raise RuntimeError("julia not found; install it with juliaup")
    return [exe, f"+{version}"] if shutil.which("juliaup") else [exe]


def run(benchmark_id, setting="setting_0"):
    t_start = time.perf_counter()
    bm = load_benchmark(benchmark_id)
    cfg = load_settings(TOOL, setting, benchmark_id)
    reason = box_to_box_reason(bm)
    if reason is not None:
        return make_result(bm, TOOL, setting, "unsupported", {"total": time.perf_counter() - t_start}, cfg,
                           detail=reason, packages=())

    work = paths.RAW_DIR / TOOL / benchmark_id / setting
    try:
        out = execute(bm, cfg, work)
    except subprocess.TimeoutExpired:
        return make_result(bm, TOOL, setting, "timeout", {"total": time.perf_counter() - t_start}, cfg,
                           detail=f"no result within {cfg['timeout_s']} s", packages=())
    except RuntimeError as exc:
        return make_result(bm, TOOL, setting, "error", {"total": time.perf_counter() - t_start}, cfg,
                           detail=str(exc), packages=())
    wall = time.perf_counter() - t_start
    verdict, detail = interpret(out)
    times = out["times"]
    runtime = {"total": wall, "compile_warmup": times["warmup_compile"], "solve": times["solve"],
               "check": times["check"]}
    params = {**cfg, "versions": out["versions"]}
    bounds = out["reach_time_bounds"] if out["horizon_covered"] else None
    extra = {k: out[k] for k in ("horizon_covered", "flowpipe_end_time", "num_segments", "num_possible", "num_inside")}
    return make_result(bm, TOOL, setting, verdict, runtime, params, detail=detail,
                       horizon_covered=out["horizon_covered"], reach_time_bounds=bounds, extra=extra, packages=())


def execute(bm, cfg, work):
    """Run run.jl on a loaded benchmark; returns its output dict."""
    work.mkdir(parents=True, exist_ok=True)
    config_path, settings_path, out_path = work / "benchmark.json", work / "settings.json", work / "result.json"
    jsonio.write_json(config_path, bm.solver_config())  # named parameters/definitions expanded
    jsonio.write_json(settings_path, cfg)
    out_path.unlink(missing_ok=True)
    cmd = julia_command(cfg["julia"]) + ["--startup-file=no", f"--project={PROJECT}",
                                         str(PROJECT / "run.jl"), str(config_path), str(settings_path), str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["timeout_s"])
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
        raise RuntimeError("julia failed: " + " | ".join(tail))
    return jsonio.read_json(out_path)


def interpret(out):
    verdict = out["verdict"]
    if verdict == "incomplete":
        verdict = "set_explosion"
        detail = f"flowpipe stopped at t = {out['flowpipe_end_time']:.4g} before the end of the horizon"
    elif verdict == "reachable_proven":
        detail = f"{out['num_inside']} flowpipe segments lie inside the target"
    elif verdict == "unreachable_proven":
        detail = f"all {out['num_segments']} flowpipe segments are disjoint from the target"
    else:
        detail = f"{out['num_possible']} of {out['num_segments']} flowpipe segments may intersect the target"
    return verdict, detail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark", nargs="+")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--setting", default="setting_0")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    ids = configured_benchmarks(TOOL, args.setting) if args.all else args.benchmark
    errors = 0
    for bid in ids:
        record = run(bid, args.setting)
        if not args.no_write:
            write_result(record)
        errors += record["verdict"] == "error"
        rt = record["runtime_s"]
        bounds = ", ".join(f"[{lo:.4g}, {hi:.4g}]" for lo, hi in (record["reach_time_bounds"] or [])[:3]) or "-"
        print(f"{bid:34s} expected={record['expected_result']:11s} {record['verdict']:20s} "
              f"solve={rt.get('solve', float('nan')):.2f}s total={rt['total']:.1f}s bounds={bounds}  {record['detail']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
