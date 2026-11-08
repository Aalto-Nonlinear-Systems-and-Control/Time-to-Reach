"""Run Hamilton-Jacobi reachability (hj_reachability) on benchmarks.

    python -m tools.hj.run --benchmark DUFF_FIN_REA_BOX [--setting setting_0]
    python -m tools.hj.run --all

The backward reachable tube of the target is computed on a grid (the
`backwards_reachable_tube` Hamiltonian postprocessor): V(x, -tau) <= 0 marks the
states whose trajectory reaches the target within time tau (for backward
benchmarks the trajectories follow -f). The tube is advanced in steps of
`time_step_report`; after each step the smallest V over samples of the initial
set is obtained with the grid's own multilinear interpolation. The verdict is
the sign of the value function as the tool returns it, with no tolerance added.
With m = the smallest V over the samples of the initial set:
  m <= 0 -> reachable_estimated    (a sampled initial state is in the tube)
  m >  0 -> unreachable_estimated
The first reported tau with a non-positive minimum is recorded as an estimate
of the earliest reach time; no reach-time bounds are claimed. (Sets of states in
the target at exactly time tau, without the tube, wash out under numerical
dissipation on thin reachable subsets such as DUFF_FIN_REA_BOX.) The target set
is given to the solver the way the level-set method takes any set: as the
values of its defining function at the grid nodes (the signed distance of a
box or a ball, the function itself for a level set within its bounding box).

Memory. A grid over a state space of four or more dimensions can need more
memory than the machine has. The computation therefore runs in its own process
under `memory_limit_gb` (tools.common.limits) and is reported as out_of_memory
  - without being started, if the node coordinates and one value array of the
    grid alone (float32) are larger than the limit, or
  - when its memory footprint exceeds the limit, or an allocation is refused.
It is reported as timeout if it has no result within `timeout_s`.
"""

import argparse
import math
import sys
import time
import traceback

import hj_reachability as hj
import jax
import jax.numpy as jnp
import numpy as np

from tools.common import load_benchmark, make_result, paths, write_result
from tools.common.limits import GB, ChildFailed, MemoryLimitExceeded, TimeLimitExceeded, run_limited
from tools.common.settings import configured_benchmarks, load_settings

from .model import Autonomous, target_values, vector_field

TOOL = "hj"
PACKAGES = ("numpy", "sympy", "jax", "jaxlib", "hj_reachability")


def grid_shape(cfg, n):
    return tuple([cfg["grid_size"]] * n) if np.isscalar(cfg["grid_size"]) else tuple(cfg["grid_size"])


def grid_memory(shape):
    """(number of nodes, bytes of the node coordinates and one value array in float32): the least a grid needs."""
    nodes = math.prod(int(k) for k in shape)  # exact: 100^16 does not fit a float mantissa or an int64
    return nodes, nodes * 4 * (len(shape) + 1)


def _analyze_benchmark(benchmark_id, cfg, save_to):
    """analyze() for a benchmark ID; the entry point of the child process (must be at module level)."""
    return analyze(load_benchmark(benchmark_id), cfg, save_to=save_to)


def run(benchmark_id, setting="setting_0"):
    t_start = time.perf_counter()
    bm = load_benchmark(benchmark_id)
    cfg = load_settings(TOOL, setting, benchmark_id)

    def result(verdict, detail, runtime=None, **kw):
        runtime = {**(runtime or {}), "total": time.perf_counter() - t_start}
        return make_result(bm, TOOL, setting, verdict, runtime, cfg, detail=detail, packages=PACKAGES, **kw)

    limit = cfg["memory_limit_gb"] * GB
    shape = grid_shape(cfg, bm.dim)
    nodes, least = grid_memory(shape)
    grid_text = (f"{shape[0]}^{len(shape)}" if len(set(shape)) == 1 else " x ".join(map(str, shape))) + \
        f" = {float(nodes):.3g} nodes"
    memory = {"memory_limit_gb": cfg["memory_limit_gb"], "grid_nodes": float(nodes), "grid_memory_gb": least / GB}
    if least > limit:
        return result("out_of_memory", f"the grid of {grid_text} needs {least / GB:.3g} GB for its node coordinates "
                      f"and one value array, above the memory limit of {cfg['memory_limit_gb']:g} GB (not started)",
                      extra={**memory, "started": False})
    try:
        (verdict, detail, runtime, extra), peak = run_limited(
            _analyze_benchmark, (benchmark_id, cfg, paths.RAW_DIR / TOOL / benchmark_id / setting),
            memory_limit=limit, timeout=cfg["timeout_s"])
    except MemoryLimitExceeded as exc:
        how = (f"an allocation was refused ({exc.note})" if exc.note else
               f"memory use reached {exc.used / GB:.3g} GB after {exc.elapsed:.3g} s")
        return result("out_of_memory", f"grid of {grid_text}: {how}, above the memory limit of "
                      f"{cfg['memory_limit_gb']:g} GB", extra={**memory, "started": True, "peak_memory_gb": exc.used / GB})
    except TimeLimitExceeded:
        return result("timeout", f"no result within {cfg['timeout_s']} s (grid of {grid_text})",
                      extra={**memory, "started": True})
    except ChildFailed as exc:
        return result("error", str(exc).strip().splitlines()[-1], extra={**memory, "started": True})
    return result(verdict, detail, runtime, horizon_covered=True,
                  extra={**extra, **memory, "peak_memory_gb": peak / GB})


def save_values(folder, grid, V, bm, taus, minima):
    folder.mkdir(parents=True, exist_ok=True)
    coords = [np.asarray(c) for c in grid.coordinate_vectors]
    data = {f"coords_{i}": c for i, c in enumerate(coords)}
    data.update(taus=taus, min_over_initial=minima)
    if V.ndim == 2:
        data["values"] = V.astype(np.float32)
    else:
        center = bm.initial_set.bounding_box().mean(axis=1)
        idx = [int(np.argmin(np.abs(c - x))) for c, x in zip(coords, center)]
        data["slice_center"] = np.array([c[k] for c, k in zip(coords, idx)])
        for i in range(V.ndim):
            for j in range(i + 1, V.ndim):
                index = tuple(slice(None) if d in (i, j) else idx[d] for d in range(V.ndim))
                data[f"slice_{i}_{j}"] = V[index].astype(np.float32)
    np.savez_compressed(folder / "values.npz", **data)


def analyze(bm, cfg, save_to=None):
    """Compute the reachable tube for a loaded benchmark; returns (verdict, detail, runtime, extra).

    With save_to (a directory), the final value function is written to values.npz for plotting:
    the whole grid in 2D, otherwise the 2D slices through the grid node nearest to the centre of
    the initial set; together with min V over the initial set at every reported time.
    """
    t_start = time.perf_counter()
    n = bm.dim
    bounds = np.asarray(cfg.get("grid_domain", bm.domain.bounds.tolist()), dtype=float)
    shape = grid_shape(cfg, n)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.asarray(bounds[:, 0]), jnp.asarray(bounds[:, 1])), shape,
        periodic_dims=tuple(cfg.get("periodic_dims", ())))
    dynamics = Autonomous(vector_field(bm.problem_system), n)
    values = target_values(bm.raw["target_sets"], n, grid)

    rng = np.random.default_rng(cfg["seed"])
    X0 = np.vstack([bm.initial_set.sample(cfg["num_samples_initial"], rng),
                    bm.initial_set.boundary_sample(cfg["boundary_samples_initial"], rng)])
    X0 = jnp.asarray(np.unique(X0, axis=0), dtype=values.dtype)
    min_over_initial = jax.jit(lambda v: jnp.min(jax.vmap(grid.interpolate, in_axes=(None, 0))(v, X0)))
    settings = hj.SolverSettings.with_accuracy(
        cfg["accuracy"], hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

    t0, t1 = bm.time_horizon
    dt = cfg["time_step_report"]
    taus = np.arange(t0, t1 + 0.5 * dt, dt)
    taus[-1] = min(taus[-1], t1)
    minima = [float(min_over_initial(values))]
    t_setup = time.perf_counter()

    t_compile = None
    for k in range(1, len(taus)):
        values = hj.step(settings, dynamics, grid, -float(taus[k - 1]), values, -float(taus[k]), progress_bar=False)
        minima.append(float(min_over_initial(values)))  # float() waits for the computation
        if t_compile is None:
            t_compile = time.perf_counter()  # first step includes JIT compilation
    t_end = time.perf_counter()

    minima = np.asarray(minima)
    if np.any(np.isnan(minima)):
        raise RuntimeError("NaN in the value function (initial set outside the grid or singular dynamics)")
    m = float(minima.min())
    if m <= 0:
        verdict, detail = "reachable_estimated", f"min V over the initial set {m:.3g} <= 0"
    else:
        verdict, detail = "unreachable_estimated", f"min V over the initial set {m:.3g} > 0"
    reached = np.flatnonzero(minima <= 0)
    if save_to is not None:
        save_values(save_to, grid, np.asarray(values), bm, taus, minima)

    runtime = {
        "setup": t_setup - t_start,
        "compile_and_first_step": (t_compile or t_end) - t_setup,
        "solve": t_end - (t_compile or t_end),
    }
    extra = {
        "direction": bm.direction,
        "grid_shape": list(shape),
        "grid_spacing": [float(s) for s in grid.spacings],
        "min_value_over_initial": m,
        "first_reach_time_estimate": float(taus[reached[0]]) if len(reached) else None,
        "num_initial_samples": int(X0.shape[0]),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    return verdict, detail, runtime, extra


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
        except Exception as exc:  # report and continue with the next benchmark
            failures += 1
            print(f"ERROR {bid}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        if not args.no_write:
            write_result(record)
        rt = record["runtime_s"]
        first = (record.get("extra") or {}).get("first_reach_time_estimate")
        first = "-" if first is None else f"{first:.4g}"
        print(f"{bid:34s} expected={record['expected_result']:11s} {record['verdict']:22s} first_reach~{first:8s} "
              f"total={rt['total']:.1f}s  {record['detail']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
