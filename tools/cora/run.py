"""Run CORA (MATLAB, submodule tools/cora/CORA at v2026.1.0) on benchmarks.

    python -m tools.cora.run --benchmark DUFF_FIN_REA_BOX [--setting setting_0]
    python -m tools.cora.run --all

Calls tools/cora/run_cora.m through `matlab -batch` and records the result.
CORA is run on what it handles itself: a forward problem from one initial box to
one target box or box shell (tools.common.support); every other benchmark is unsupported.
Verdicts as for JuliaReach:
  some reachable set lies inside the target     -> reachable_proven
  horizon covered, every set disjoint           -> unreachable_proven
  horizon covered, some set may intersect       -> reachable_possible (no proof:
                                                   the sets are over-approximations)
  reachable set stops before the horizon ends   -> set_explosion
  no result within timeout_s                    -> timeout
MATLAB is found through $MATLAB_BIN, /Applications/MATLAB_<release>.app on
macOS, or `matlab` on the PATH.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tools.common import jsonio, load_benchmark, make_result, paths, write_result
from tools.common.settings import configured_benchmarks, load_settings
from tools.common.support import box_to_box_reason

TOOL = "cora"
HERE = paths.REPO_ROOT / "tools" / "cora"


def matlab_command(release):
    if os.environ.get("MATLAB_BIN"):
        return os.environ["MATLAB_BIN"]
    app = Path(f"/Applications/MATLAB_{release}.app/bin/matlab")
    if platform.system() == "Darwin" and app.exists():
        return str(app)
    exe = shutil.which("matlab")
    if exe is None:
        raise RuntimeError(f"MATLAB {release} not found; set MATLAB_BIN")
    return exe


def run_matlab(release, call, work, timeout):
    """`matlab -batch call` with its output in work/matlab.log; returns (exit code, log text).

    The output goes to a file, not a pipe: MATLAB may start background services
    (MathWorksServiceHost) that inherit its stdout/stderr and keep a pipe open after
    MATLAB exits, which would block until the timeout. Raises TimeoutExpired.
    """
    log = work / "matlab.log"
    with log.open("w") as f:
        proc = subprocess.run([matlab_command(release), "-batch", call], stdin=subprocess.DEVNULL,
                              stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode, log.read_text(errors="replace")


def execute(bm, cfg, work):
    """Run run_cora.m on a loaded benchmark; returns its output dict."""
    work.mkdir(parents=True, exist_ok=True)
    config_path, settings_path, out_path = work / "benchmark.json", work / "settings.json", work / "result.json"
    jsonio.write_json(config_path, bm.solver_config())  # named parameters/definitions expanded
    jsonio.write_json(settings_path, cfg)
    out_path.unlink(missing_ok=True)
    call = f"addpath('{HERE}'); run_cora('{config_path}', '{settings_path}', '{out_path}')"
    code, log = run_matlab(cfg["matlab_release"], call, work, cfg["timeout_s"])
    if code != 0 or not out_path.exists():
        tail = log.strip().splitlines()[-6:]
        raise RuntimeError("MATLAB failed: " + " | ".join(tail))
    return jsonio.read_json(out_path)


def interpret(out):
    verdict = out["verdict"]
    if verdict == "incomplete":
        verdict = "set_explosion"
        t_end = out["reachable_set_end_time"]
        detail = (f"reachable set stops at t = {t_end:.4g} before the end of the horizon" if t_end is not None
                  else "reachable set explosion at the first step")
    elif verdict == "reachable_proven":
        detail = f"{out['num_inside']} reachable sets lie inside the target"
    elif verdict == "unreachable_proven":
        detail = f"all {out['num_segments']} reachable sets are disjoint from the target"
    else:
        detail = f"{out['num_possible']} of {out['num_segments']} reachable sets may intersect the target"
    return verdict, detail


def _bounds(raw):
    # jsonencode writes one [a, b] row as a flat list
    if not raw:
        return []
    return [raw] if not isinstance(raw[0], list) else raw


def run(benchmark_id, setting="setting_0"):
    t_start = time.perf_counter()
    bm = load_benchmark(benchmark_id)
    cfg = load_settings(TOOL, setting, benchmark_id)
    reason = box_to_box_reason(bm)
    if reason is not None:
        return make_result(bm, TOOL, setting, "unsupported", {"total": time.perf_counter() - t_start}, cfg,
                           detail=reason, packages=())
    try:
        out = execute(bm, cfg, paths.RAW_DIR / TOOL / benchmark_id / setting)
    except subprocess.TimeoutExpired:
        return make_result(bm, TOOL, setting, "timeout", {"total": time.perf_counter() - t_start}, cfg,
                           detail=f"no result within {cfg['timeout_s']} s", packages=())
    except RuntimeError as exc:
        return make_result(bm, TOOL, setting, "error", {"total": time.perf_counter() - t_start}, cfg,
                           detail=str(exc), packages=())
    verdict, detail = interpret(out)
    times = out["times"]
    runtime = {"total": time.perf_counter() - t_start, "warmup": times["warmup"], "reach": times["reach"],
               "check": times["check"]}
    params = {**cfg, "versions": out["versions"]}
    bounds = _bounds(out["reach_time_bounds"]) if out["horizon_covered"] else None
    extra = {k: out[k] for k in ("horizon_covered", "reachable_set_end_time", "num_segments", "num_possible", "num_inside")}
    return make_result(bm, TOOL, setting, verdict, runtime, params, detail=detail,
                       horizon_covered=out["horizon_covered"], reach_time_bounds=bounds, extra=extra, packages=())


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
              f"reach={rt.get('reach', float('nan')):.2f}s total={rt['total']:.1f}s bounds={bounds}  {record['detail']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
