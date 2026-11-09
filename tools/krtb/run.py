"""Run KRTB (Koopman reach-time bounds) on benchmarks.

    python -m tools.krtb.run --benchmark DUFF_FIN_REA_BOX [--setting setting_0]
    python -m tools.krtb.run --all

For each benchmark: sample the initial and target sets (a box-shell target around
the equilibrium also gets the point of its inner faces, see equilibrium_sample), evaluate the principal
eigenfunctions (analytic or path integral), compute reach-time bounds (with
negated eigenvalues for backward benchmarks) clipped to the time horizon, and
look for a simulated witness inside the bounds. Verdicts:
  - empty bounds                      -> unreachable_estimated (eigenfunction
                                         ranges come from samples, so this is
                                         not a proof)
  - bounds and a witness inside them  -> reachable_witness
  - bounds but no witness found       -> unknown
  - no information from the bounds    -> unknown
Results go to results/summary/krtb/<ID>/<setting>.json. Runtime phases:
setup (sampling), eigenpairs (principal eigenvalues and eigenfunction values on
the samples), bounds (reach-time bounds), witness_search; krtb = eigenpairs +
bounds.
"""

import argparse
import sys
import time
import traceback

import numpy as np

from tools.common import load_benchmark, make_result, write_result
from tools.common.sets import BoxShell
from tools.common.settings import configured_benchmarks, load_settings
from tools.common.simulate import check_trajectory

from . import eigenfunctions
from .bounds import compute_reach_time_bounds

TOOL = "krtb"
PACKAGES = ("numpy", "scipy", "sympy", "cvxpy", "Mosek")


def sample(s, n_interior, n_boundary, rng, n_vertex=0):
    """Uniform interior samples, plus samples on the boundary and towards the vertices (extrema of log|psi| lie there)."""
    parts = [s.sample(n_interior, rng)]
    if n_boundary > 0:
        parts.append(s.boundary_sample(n_boundary, rng))
    if n_vertex > 0:
        parts.append(s.vertex_sample(n_vertex, rng))
    return np.vstack(parts)


def equilibrium_sample(target, equilibrium):
    """The point of a box-shell target nearest to the equilibrium x_e, or None for other targets.

    A target around x_e is a box shell (tools.common.sets.BoxShell): x_e itself is excluded, as the bounds
    assume. Its inner faces, at the distance inner_radius from x_e, are where the principal eigenfunctions are
    smallest on the target: every principal eigenfunction vanishes at x_e (s_t(x_e) = x_e gives
    psi(x_e) = e^{lambda t} psi(x_e) for all t). The infimum of |psi| on the target, which sets the upper end of
    the reach-time bound, therefore lies there. Uniform samples do not find it: they approach the zero set of a
    complex psi (of codimension 2) only like N^(-1/2), so every tenfold increase of the samples adds only about
    ln(10) / (2 |Re lambda|) to the sampled upper end. On the cart-pole convergence benchmark (fast mode
    Re lambda = -3.62, horizon 12) the upper end moves from 3.51 with 16 000 samples to 3.99-4.12 with
    511 000, while the true one lies beyond the horizon. One point of the inner faces gives it.

    The point is x_e + inner_radius * u, with u (|u|_inf = 1) from its own fixed generator, so the other
    samples and the witness search draw the same random numbers as without this point.
    """
    if equilibrium is None or not isinstance(target, BoxShell):
        return None
    x_e = np.asarray(equilibrium, float)
    if not np.array_equal(target.center, x_e):
        return None
    u = np.random.default_rng(0).uniform(-1.0, 1.0, x_e.size)
    u /= np.abs(u).max()
    for x in (x_e + target.inner_radius * u, x_e - target.inner_radius * u):
        if np.atleast_1d(target.contains(x[None]))[0]:
            return x
    return None


def build_eigenfunctions(bm, spec, pi_defaults, rng):
    if spec["method"] == "analytic":
        check = bm.domain.sample(200, rng)
        return eigenfunctions.analytic(spec["functions"], spec["eigenvalues"], bm.system, check_points=check)
    if spec["method"] == "path_integral":
        opts = {**pi_defaults, **{k: v for k, v in spec.items() if k in ("rtol", "atol", "batch", "dt")}}
        return eigenfunctions.path_integral(bm.system, spec["equilibrium"], spec["T"], **opts)
    raise ValueError(f"unknown eigenfunction method {spec['method']!r}")


def find_witness(bm, intervals, n, rng):
    """Simulate up to n initial points (half on the boundary, where reaching subsets of
    the initial set often lie) up to the end of the last interval; first hit wins."""
    t_end = max(hi for _, hi in intervals)
    horizon = (bm.time_horizon[0], t_end)
    X0 = np.unique(np.vstack([bm.initial_set.sample(n - n // 2, rng), bm.initial_set.boundary_sample(n // 2, rng)]), axis=0)
    for x0 in X0:
        r = check_trajectory(bm.problem_system, x0, bm.target_set, horizon)
        if r.hit:
            inside = any(lo - 1e-9 <= r.first_hit_time <= hi + 1e-9 for lo, hi in intervals)
            return {"x0": x0, "first_hit_time": r.first_hit_time, "inside_bounds": inside}
    return None


def run(benchmark_id, setting="setting_0"):
    t_start = time.perf_counter()
    bm = load_benchmark(benchmark_id)
    cfg = load_settings(TOOL, setting, benchmark_id)
    rng = np.random.default_rng(cfg["seed"])

    X0 = sample(bm.initial_set, cfg["num_samples_initial"], cfg["boundary_samples_initial"], rng,
                cfg.get("vertex_samples_initial", 0))
    XF = sample(bm.target_set, cfg["num_samples_target"], cfg["boundary_samples_target"], rng,
                cfg.get("vertex_samples_target", 0))
    # a box shell around the equilibrium gets the point of its inner faces instead of a huge sample (see equilibrium_sample)
    x_eq = equilibrium_sample(bm.target_set, cfg["eigenfunctions"].get("equilibrium"))
    if x_eq is not None:
        XF = np.vstack([XF, x_eq])
    t_setup = time.perf_counter()

    # principal eigenpairs: the eigenvalues and the eigenfunction values on the samples
    eig = build_eigenfunctions(bm, cfg["eigenfunctions"], cfg["path_integral"], rng)
    ef0, efF = eig.evaluate(X0), eig.evaluate(XF)
    if x_eq is not None and not np.all(np.abs(efF[-1]) > 0):  # psi underflowed to 0 (log 0): drop the point
        XF, efF, x_eq = XF[:-1], efF[:-1], None
    t_eig = time.perf_counter()

    # The eigenfunctions of -f are those of f with negated eigenvalues.
    lams = eig.eigenvalues if bm.direction == "forward" else -eig.eigenvalues
    bounds = compute_reach_time_bounds(ef0, efF, lams, horizon=bm.time_horizon)
    t_bounds = time.perf_counter()

    witness = None
    if bounds.status == "no_information":
        verdict, detail = "unknown", "the eigenfunction ranges give no information on the reach time"
    elif bounds.empty:
        verdict, detail = "unreachable_estimated", "reach-time bounds do not meet the time horizon"
    else:
        witness = find_witness(bm, bounds.intervals, cfg["num_sim_samples"], rng)
        if witness is None:
            verdict, detail = "unknown", "bounds meet the horizon but no simulated witness was found"
        else:
            verdict = "reachable_witness"
            detail = f"witness reaches the target at t = {witness['first_hit_time']:.4g}"
            if not witness["inside_bounds"]:
                detail += " (outside the computed bounds)"
    t_end = time.perf_counter()

    extra = {
        "direction": bm.direction,
        "eigenvalues": [[float(np.real(v)), float(np.imag(v))] for v in eig.eigenvalues],
        "eigenfunctions": eig.info,
        "bounds_status": bounds.status,
        "bounds_unclipped": bounds.unclipped,
        "magnitude_interval": bounds.magnitude,
        "phase_window": bounds.phase,
        "period": bounds.period,
        "bound_details": bounds.details,
        "witness": witness,
        "num_samples": {"initial": len(X0), "target": len(XF)},
        "equilibrium_sample": None if x_eq is None else x_eq.tolist(),
    }
    params = {k: v for k, v in cfg.items()}
    runtime = {
        "total": t_end - t_start,
        "setup": t_setup - t_start,
        "eigenpairs": t_eig - t_setup,
        "bounds": t_bounds - t_eig,
        "krtb": t_bounds - t_setup,
        "witness_search": t_end - t_bounds,
    }
    return make_result(
        bm, TOOL, setting, verdict, runtime, params, detail=detail, horizon_covered=True,
        reach_time_bounds=bounds.intervals, extra=extra, packages=PACKAGES,
    )


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
        bounds = ", ".join(f"[{lo:.4g}, {hi:.4g}]" for lo, hi in record["reach_time_bounds"]) or "-"
        print(f"{bid:34s} expected={record['expected_result']:11s} {record['verdict']:22s} "
              f"bounds={bounds:28s} eigenpairs={rt['eigenpairs']:.2f}s bounds_time={rt['bounds']:.3f}s "
              f"witness={rt['witness_search']:.2f}s total={rt['total']:.1f}s  {record['detail']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
