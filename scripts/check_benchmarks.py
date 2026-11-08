"""Validate benchmark configs and check their expected results by simulation.

For every benchmark this script
  1. validates the config against configs/schema/benchmark.schema.json and builds its sets,
  2. checks that the initial and target sets lie inside the domain and are non-empty,
  3. simulates trajectories from interior and boundary samples of the initial set with
     dense in-step sampling and root finding on the target's inside function, and records
       - for 'reachable' benchmarks: a witness (x0, first hit time), how many samples hit,
         and the simulated in-target time set: the union over sampled trajectories of the
         times at which they are inside the target (a sampled under-approximation of the
         true set, used to judge reach-time bounds of the tools),
       - for 'unreachable' benchmarks: the closest approach to the target (no counterexample
         found is evidence, not a proof),
       - whether trajectories stay inside the domain over the horizon,
  4. writes results/summary/ground_truth/<ID>/simulation.json.

The exit code is non-zero if any benchmark fails a check, so this can run in CI.

Usage (from the repo root):
    python -m scripts.check_benchmarks                 # all benchmarks
    python -m scripts.check_benchmarks DUFF_FIN_REA_BOX NL_SLC_FIN_UNR_BOX
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tools.common import list_benchmarks, load_benchmark, make_result, write_result
from tools.common.sets import Box, LevelSet, Union
from tools.common.simulate import check_trajectory, merge_intervals

DOMAIN_TOL = 1e-9

_worker_benchmark = None


def _init_worker(benchmark_id):
    global _worker_benchmark
    _worker_benchmark = load_benchmark(benchmark_id)


def _check_one(x0):
    bm = _worker_benchmark
    return check_trajectory(bm.problem_system, x0, bm.target_set, bm.time_horizon, domain=bm.domain)


def initial_samples(bm, n_interior, n_boundary, seed):
    rng = np.random.default_rng(seed)
    X = np.vstack([bm.initial_set.sample(n_interior, rng), bm.initial_set.boundary_sample(n_boundary, rng)])
    return np.unique(X, axis=0)


def set_checks(bm):
    """Problems with the sets themselves (empty, outside the domain)."""
    problems, info = [], []
    rng = np.random.default_rng(0)
    for role, s in (("initial", bm.initial_set), ("target", bm.target_set)):
        parts = s.sets if isinstance(s, Union) else [s]
        for i, part in enumerate(parts):
            name = f"{role} set {i + 1} ({part.kind})"
            bb = part.bounding_box()
            if np.any(bb[:, 0] < bm.domain.lower - DOMAIN_TOL) or np.any(bb[:, 1] > bm.domain.upper + DOMAIN_TOL):
                problems.append(f"{name}: bounding box {bb.tolist()} is not inside the domain")
            try:
                part.sample(10, rng)
            except ValueError as exc:
                problems.append(f"{name}: {exc}")
            if isinstance(part, LevelSet) and part.boundary_touches_box(rng=rng):
                info.append(f"{name}: the sublevel set reaches its bounding box (the box cuts the set)")
    return problems, info


def check_benchmark(benchmark_id, n_interior, n_boundary, seed, jobs):
    t_start = time.perf_counter()
    bm = load_benchmark(benchmark_id)
    problems, info = set_checks(bm)

    X0 = initial_samples(bm, n_interior, n_boundary, seed)
    if jobs > 1 and len(X0) > 1:
        with ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(benchmark_id,)) as ex:
            runs = list(ex.map(_check_one, X0, chunksize=max(1, len(X0) // (4 * jobs))))
    else:
        _init_worker(benchmark_id)
        runs = [_check_one(x0) for x0 in X0]

    failed = [r for r in runs if not r.success]
    if failed:
        problems.append(f"{len(failed)} integrations failed, e.g. x0={failed[0].x0.tolist()}: {failed[0].message}")
    ok = [r for r in runs if r.success]
    hits = [r for r in ok if r.hit]
    min_g = min(r.min_g for r in ok)
    closest = min(ok, key=lambda r: r.min_g)
    domain_violation = max(r.max_domain_violation for r in ok)
    if domain_violation > DOMAIN_TOL:
        worst = max(ok, key=lambda r: r.max_domain_violation)
        problems.append(f"trajectories leave the domain by up to {domain_violation:.4g} (x0={worst.x0.tolist()})")

    extra = {
        "samples": {"interior": n_interior, "boundary": n_boundary, "unique_initial_points": len(X0), "seed": seed},
        "num_hitting": len(hits),
        "num_trajectories": len(ok),
        "closest_approach": {"value": min_g, "kind": bm.target_set.margin_kind(), "x0": closest.x0, "time": closest.min_g_time},
        "max_domain_violation": domain_violation,
        "problems": problems,
        "info": info,
    }
    if hits:
        first = min(hits, key=lambda r: r.first_hit_time)
        extra["witness"] = {"x0": first.x0, "first_hit_time": first.first_hit_time}
        extra["first_hit_time_range"] = [min(r.first_hit_time for r in hits), max(r.first_hit_time for r in hits)]
        extra["in_target_times"] = merge_intervals([iv for r in hits for iv in r.in_target], tol=1e-9)
        verdict, detail = "reachable_witness", f"{len(hits)}/{len(ok)} sampled trajectories reach the target"
    else:
        verdict = "unreachable_estimated"
        detail = f"no sampled trajectory reaches the target; closest approach {min_g:.4g} ({bm.target_set.margin_kind()})"

    record = make_result(
        bm, "ground_truth", "simulation", verdict,
        runtime_s={"total": time.perf_counter() - t_start},
        params={"direction": bm.direction, "integrator": "DOP853", "rtol": 1e-9, "atol": 1e-11, "event_detection": True},
        detail=detail, horizon_covered=True, extra=extra,
    )
    passed = record["agrees_with_expected"] and not problems
    return record, passed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmarks", nargs="*", help="benchmark IDs (default: all)")
    parser.add_argument("--interior", type=int, default=1000, help="interior samples of the initial set")
    parser.add_argument("--boundary", type=int, default=1000, help="boundary samples of the initial set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--no-write", action="store_true", help="do not write result summaries")
    args = parser.parse_args(argv)

    ids = args.benchmarks or list_benchmarks()
    all_passed = True
    for bid in ids:
        try:
            record, passed = check_benchmark(bid, args.interior, args.boundary, args.seed, args.jobs)
        except Exception as exc:  # report and continue with the next benchmark
            print(f"ERROR {bid}: {type(exc).__name__}: {exc}")
            all_passed = False
            continue
        all_passed &= passed
        if not args.no_write:
            write_result(record)
        ex = record["extra"]
        status = "PASS" if passed else "FAIL"
        line = f"{status} {bid:34s} expected={record['expected_result']:11s} {record['verdict']:22s} {record['detail']}"
        if "witness" in ex:
            lo, hi = ex["first_hit_time_range"]
            line += f"; first hits in [{lo:.4g}, {hi:.4g}]"
        print(line + f"  ({record['runtime_s']['total']:.1f}s)")
        for p in ex["problems"]:
            print(f"      problem: {p}")
        for i in ex["info"]:
            print(f"      info: {i}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
