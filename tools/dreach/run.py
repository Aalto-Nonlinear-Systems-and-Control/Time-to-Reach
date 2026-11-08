"""Run dReach (dReal3) on benchmarks inside Docker.

    python -m tools.dreach.run --benchmark DUFF_FIN_REA_BOX [--setting setting_0]
    python -m tools.dreach.run --all

The model is generated from the config (tools/dreach/gen_drh.py) and dReach runs
with -k 0 (no jumps) and -z (exit codes 51 = delta-sat, 52 = unsat). Verdicts:
  unsat      -> unreachable_proven  (sound)
  delta-sat  -> reachable_possible  (reachable up to a delta-perturbation of the
                                     problem; not a proof of reachability)
  timeout    -> timeout
  otherwise  -> error
Build the image first: docker build --platform linux/amd64 -t <image> tools/dreach
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid

from tools.common import load_benchmark, make_result, paths, write_result
from tools.common.settings import configured_benchmarks, load_settings

from .gen_drh import generate_model

TOOL = "dreach"


def docker_cli():
    for name in ("docker", "podman"):
        if shutil.which(name):
            return name
    raise RuntimeError("neither docker nor podman is installed")


def image_id(cli, image):
    out = subprocess.run([cli, "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True)
    if out.returncode != 0:
        info = subprocess.run([cli, "info"], capture_output=True, text=True)
        if info.returncode != 0:
            raise RuntimeError(f"{cli} daemon is not running (on macOS: start OrbStack or Docker Desktop)")
        raise RuntimeError(f"image {image!r} not found; build it with: {cli} build --platform linux/amd64 "
                           f"-t {image} tools/dreach")
    return out.stdout.strip()


def execute(bm, cfg, work):
    """Run dReach in Docker on a loaded benchmark; returns the raw outcome."""
    t_start = time.perf_counter()
    cli = docker_cli()
    img = image_id(cli, cfg["image"])

    work.mkdir(parents=True, exist_ok=True)
    for name in ("dreach.out", "dreach.err", "dreach.code"):
        (work / name).unlink(missing_ok=True)
    model = f"{bm.id}.drh"
    (work / model).write_text(generate_model(bm))

    container = f"dreach-{bm.id.lower()}-{uuid.uuid4().hex[:8]}"
    inner = (f"/usr/bin/time -p dReach -k {cfg['k']} -z {model} "
             f"--precision {cfg['precision']} > dreach.out 2> dreach.err; echo $? > dreach.code")
    # run as the host user so that files written to the mounted work directory are not owned by root
    cmd = [cli, "run", "--rm", "--name", container, "--platform", "linux/amd64",
           "--user", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{work}:/work", "-w", "/work", cfg["image"], "bash", "-c", inner]

    timed_out = False
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["timeout_s"])
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run([cli, "kill", container], capture_output=True)
    wall = time.perf_counter() - t_start

    out = (work / "dreach.out").read_text() if (work / "dreach.out").exists() else ""
    err = (work / "dreach.err").read_text() if (work / "dreach.err").exists() else ""
    code = (work / "dreach.code").read_text().strip() if (work / "dreach.code").exists() else None
    real = re.search(r"^real\s+([\d.]+)", err, re.M)
    return {"timed_out": timed_out, "code": code, "stdout": out, "stderr": err, "wall": wall,
            "dreach_time": float(real.group(1)) if real else None, "image_id": img, "model": work / model}


def interpret(out, cfg):
    """Verdict and detail from the outcome of execute()."""
    if out["timed_out"]:
        return "timeout", f"no answer within {cfg['timeout_s']} s"
    if out["code"] == "52":
        return "unreachable_proven", "unsat"
    if out["code"] == "51":
        return "reachable_possible", f"delta-sat with delta = {cfg['precision']}"
    tail = (out["stdout"] + out["stderr"]).strip().splitlines()[-3:]
    return "error", f"dReach exit code {out['code']}: " + " | ".join(tail)


def run(benchmark_id, setting="setting_0"):
    bm = load_benchmark(benchmark_id)
    cfg = load_settings(TOOL, setting, benchmark_id)
    out = execute(bm, cfg, paths.RAW_DIR / TOOL / benchmark_id / setting)
    verdict, detail = interpret(out, cfg)

    host = platform.machine()
    params = {**cfg, "image_id": out["image_id"], "host_machine": host, "emulated": host not in ("x86_64", "AMD64")}
    dreach_time = out["dreach_time"]
    runtime = {"total": out["wall"], "dreach": dreach_time,
               "container_overhead": None if dreach_time is None else out["wall"] - dreach_time}
    extra = {"model": str(out["model"]), "stdout_tail": out["stdout"].strip().splitlines()[-5:]}
    return make_result(bm, TOOL, setting, verdict, runtime, params, detail=detail,
                       horizon_covered=verdict in ("unreachable_proven", "reachable_possible"),
                       extra=extra, packages=("numpy",))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark", nargs="+")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--setting", default="setting_0")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    ids = configured_benchmarks(TOOL, args.setting) if args.all else args.benchmark
    failures = 0
    for bid in ids:
        try:
            record = run(bid, args.setting)
        except RuntimeError as exc:
            print(f"ERROR {bid}: {exc}")
            return 2
        if not args.no_write:
            write_result(record)
        failures += record["verdict"] == "error"
        rt = record["runtime_s"]
        print(f"{bid:34s} expected={record['expected_result']:11s} {record['verdict']:20s} "
              f"dreach={rt['dreach']}s total={rt['total']:.1f}s  {record['detail']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
