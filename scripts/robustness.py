"""Reproduce the robustness study of the paper with the current KRTB implementation
(paragraph "Robustness to eigenfunction estimation errors" of the Duffing example and
its figure, the distribution of the reach-time bounds with inaccurate eigenfunctions).

    python -m scripts.robustness [--eps 0.01 0.001] [--trials 3000] [--seed 0] [--benchmark ID]
                                 [--samples 1000] [--setting setting_0] [--no-write]

This experiment is separate from the benchmark comparison: it is not run by the tool
runners, and its output goes to results/robustness/, not to results/summary/.

Samples. The two sets of the Duffing example with non-convex sets are sampled as in the
original experiment (--samples: 1000 uniform points in each set). The eigenfunctions
and the error-free reference are those of the benchmark comparison (--setting: the KRTB
setting, which also samples the boundaries of the sets): the reference is its magnitude
interval. So that the samples of the experiment have exactly this interval when there
is no noise, the points of the comparison's data at which each |psi_i| is largest and
smallest on each set are added to them (two points per set here). With --samples 0 the
samples of the KRTB setting themselves are used.

Noise. In every trial each value psi_i(x) is replaced by (1 + delta(x)) psi_i(x), with
delta(x) drawn uniformly from [-eps, eps] independently for every sample point x and
shared by all i, and the magnitude interval is computed with
tools.krtb.bounds.magnitude_bound, unchanged. The factor 1 + delta(x) is real and
positive, so the phases are not affected.

Since log(1 - eps) <= log(1 + delta) <= log(1 + eps), every extremum of log|psi_i| over
a set moves by at most that much, and with eigenvalues whose real parts all have the
same sign each end of the magnitude interval moves by at most

    margin = log((1 + eps) / (1 - eps)) / min_i |Re(lam_i)|.

Widening the computed interval by this margin therefore contains the error-free one.

Writes results/robustness/<benchmark>.json (statistics and the bounds of every trial)
and results/robustness/<name>.png and .pdf (histograms; dashed line: error-free).
"""

import argparse
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from tools.common import jsonio, load_benchmark, paths  # noqa: E402
from tools.common.results import provenance, summary_path  # noqa: E402
from tools.common.settings import load_settings  # noqa: E402
from tools.krtb.bounds import magnitude_bound  # noqa: E402
from tools.krtb.run import PACKAGES, TOOL, build_eigenfunctions, sample  # noqa: E402

DEFAULT_BENCHMARK = "DUFF_FIN_REA_NONCONVEX_LEVEL_SET"
DEFAULT_SETTING = "setting_0"  # the KRTB setting of the comparison: eigenfunctions and the error-free reference
DEFAULT_SAMPLES = 1000  # the sampling of the original experiment: uniform points inside each set, none on the boundary
OUT_DIR = paths.RESULTS_DIR / "robustness"
FIGURE_NAMES = {DEFAULT_BENCHMARK: "bound_noise_nonconvex"}  # file name used by the paper


def eigenfunction_values(benchmark_id, setting, num_samples=None):
    """Eigenfunction values on the samples of the two sets and the eigenvalues, as in tools.krtb.run.run.

    The samples are those of the KRTB setting, or with num_samples that many uniform points inside each set.
    """
    bm = load_benchmark(benchmark_id)
    cfg = load_settings(TOOL, setting, benchmark_id)
    if num_samples is not None:
        cfg = {**cfg, "num_samples_initial": num_samples, "num_samples_target": num_samples,
               "boundary_samples_initial": 0, "boundary_samples_target": 0,
               "vertex_samples_initial": 0, "vertex_samples_target": 0}
    rng = np.random.default_rng(cfg["seed"])
    X0 = sample(bm.initial_set, cfg["num_samples_initial"], cfg["boundary_samples_initial"], rng,
                cfg.get("vertex_samples_initial", 0))
    XF = sample(bm.target_set, cfg["num_samples_target"], cfg["boundary_samples_target"], rng,
                cfg.get("vertex_samples_target", 0))
    eig = build_eigenfunctions(bm, cfg["eigenfunctions"], cfg["path_integral"], rng)
    lams = eig.eigenvalues if bm.direction == "forward" else -eig.eigenvalues
    return eig.evaluate(X0), eig.evaluate(XF), np.asarray(lams, dtype=complex)


def extremal_rows(ef):
    """The rows (sample points) at which the modulus of some eigenfunction is largest or smallest."""
    modulus = np.abs(ef)
    return ef[sorted({int(i) for i in np.concatenate([modulus.argmax(axis=0), modulus.argmin(axis=0)])})]


def perturb(ef, eps, rng):
    """(1 + delta(x)) psi_i(x): one delta in [-eps, eps] per sample point (row), shared by all eigenfunctions."""
    return (1.0 + rng.uniform(-eps, eps, size=(ef.shape[0], 1))) * ef


def error_margin(eps, lams):
    """Largest shift of an end of the magnitude interval, or None if the real parts change sign."""
    re = np.real(lams)
    if not (np.all(re > 0) or np.all(re < 0)):
        return None
    return float(np.log((1 + eps) / (1 - eps)) / np.abs(re).min())


def run_trials(ef0, efF, lams, eps, trials, rng):
    """Lower and upper ends of the magnitude interval in every trial (nan where the interval is empty)."""
    out = np.full((trials, 2), np.nan)
    for k in range(trials):
        mag, _ = magnitude_bound(perturb(ef0, eps, rng), perturb(efF, eps, rng), lams)
        if mag is not None:
            out[k] = mag
    return out


def statistics(bounds, exact, margin):
    """Summary of the trials against the error-free interval `exact`."""
    ok = bounds[~np.isnan(bounds).any(axis=1)]
    lo, hi = ok[:, 0], ok[:, 1]
    stats = {
        "num_trials": int(len(bounds)),
        "num_empty": int(len(bounds) - len(ok)),
        "lower": {"mean": lo.mean(), "std": lo.std(), "min": lo.min(), "max": lo.max()},
        "upper": {"mean": hi.mean(), "std": hi.std(), "min": hi.min(), "max": hi.max()},
        "max_abs_shift": {"lower": np.abs(lo - exact[0]).max(), "upper": np.abs(hi - exact[1]).max()},
        # the perturbed interval contains / lies inside the error-free one
        "fraction_containing_exact": float(np.mean((lo <= exact[0]) & (hi >= exact[1]))),
        "fraction_inside_exact": float(np.mean((lo >= exact[0]) & (hi <= exact[1]))),
        "margin": margin,
    }
    if margin is not None:
        stats["fraction_within_margin"] = float(np.mean((np.abs(lo - exact[0]) <= margin) & (np.abs(hi - exact[1]) <= margin)))
    return stats


def experiment(benchmark_id=DEFAULT_BENCHMARK, setting=DEFAULT_SETTING, num_samples=DEFAULT_SAMPLES,
               eps_levels=(0.01, 0.001), trials=3000, seed=0):
    """num_samples uniform points inside each set plus the extremal points of the KRTB setting's data;
    num_samples=None: the samples of the KRTB setting themselves."""
    t0 = time.perf_counter()
    ref0, refF, lams = eigenfunction_values(benchmark_id, setting)
    exact, _ = magnitude_bound(ref0, refF, lams)
    if exact is None:
        raise ValueError(f"{benchmark_id}: the error-free magnitude interval is empty")
    if num_samples is None:
        ef0, efF, added = ref0, refF, {"initial": 0, "target": 0}
    else:
        ef0, efF, _ = eigenfunction_values(benchmark_id, setting, num_samples)
        extra0, extraF = extremal_rows(ref0), extremal_rows(refF)
        ef0, efF = np.vstack([ef0, extra0]), np.vstack([efF, extraF])
        added = {"initial": int(len(extra0)), "target": int(len(extraF))}
    own, _ = magnitude_bound(ef0, efF, lams)
    if own is None or not np.allclose(own, exact, atol=1e-9, rtol=0):
        raise ValueError(f"{benchmark_id}: {num_samples} uniform points per set with the extremal points of {setting} "
                         f"give {own}, not the reference interval {exact}; the data of {setting} must reach further "
                         f"than the samples")
    record = {
        "schema_version": 2,
        "experiment": "robustness",
        "benchmark": benchmark_id,
        "tool": TOOL,
        "setting": setting,
        "uniform_samples_per_set": num_samples,
        "noise_model": "psi_i(x) -> (1 + delta(x)) psi_i(x), delta(x) ~ U(-eps, eps) per sample point, shared by all i",
        "quantity": "magnitude interval (tools.krtb.bounds.magnitude_bound)",
        "seed": seed,
        "num_samples": {"initial": int(len(ef0)), "target": int(len(efF))},
        "num_added_extremal_points": added,
        "eigenvalues": [[float(v.real), float(v.imag)] for v in lams],
        "error_free": [float(exact[0]), float(exact[1])],
        "matches_stored_result": matches_stored(benchmark_id, setting, exact),
        "levels": [],
    }
    for i, eps in enumerate(eps_levels):
        bounds = run_trials(ef0, efF, lams, eps, trials, np.random.default_rng([seed, i]))
        level = {"eps": eps, **statistics(bounds, exact, error_margin(eps, lams))}
        level["lower_bounds"] = np.round(bounds[:, 0], 6)
        level["upper_bounds"] = np.round(bounds[:, 1], 6)
        record["levels"].append(level)
    record["runtime_s"] = time.perf_counter() - t0
    record["provenance"] = provenance(PACKAGES)
    return record


def matches_stored(benchmark_id, setting, exact, tol=1e-9):
    """Whether the error-free interval equals the one stored by tools.krtb.run (None if there is no stored result)."""
    path = summary_path(TOOL, benchmark_id, setting)
    if not path.exists():
        return None
    stored = jsonio.read_json(path)["extra"]["magnitude_interval"]
    return bool(stored is not None and np.allclose(stored, exact, atol=tol, rtol=0))


def figure(record, bins=50):
    """Histograms of the two ends of the magnitude interval, one row per noise level."""
    levels = record["levels"]
    fig, axes = plt.subplots(len(levels), 2, figsize=(10, 2.5 * len(levels)), squeeze=False)
    for row, level in zip(axes, levels):
        for ax, key, name, exact in zip(row, ("lower_bounds", "upper_bounds"), ("Lower", "Upper"), record["error_free"]):
            values = np.asarray(level[key], dtype=float)
            ax.hist(values[~np.isnan(values)], bins=bins, color="#7fb0ea")
            ax.axvline(exact, color="#e8443a", linestyle="--", linewidth=1.5)
            ax.set_title(rf"{name} Bound ($\varepsilon={level['eps']:g}$)", fontsize=20)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # few ticks, so that the labels do not overlap
            ax.tick_params(labelsize=12)
        row[0].set_ylabel("Frequency", fontsize=20)
    for ax in axes[-1]:
        ax.set_xlabel("Time", fontsize=20)
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--setting", default=DEFAULT_SETTING,
                        help="KRTB setting: the eigenfunctions, and the error-free reference (its magnitude interval)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="uniform points inside each set (0: the samples of the KRTB setting)")
    parser.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.001], help="noise levels")
    parser.add_argument("--trials", type=int, default=3000, help="trials per noise level")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    record = experiment(args.benchmark, args.setting, args.samples or None, args.eps, args.trials, args.seed)
    print(f"{args.benchmark}: {record['num_samples']} samples, of which {record['num_added_extremal_points']} are "
          f"extremal points of KRTB {args.setting}; error-free magnitude interval [{record['error_free'][0]:.4f}, "
          f"{record['error_free'][1]:.4f}], equal to the stored KRTB result: {record['matches_stored_result']}")
    for level in record["levels"]:
        lo, hi = level["lower"], level["upper"]
        print(f"  eps = {level['eps']:g}: lower {lo['mean']:.4f} +- {lo['std']:.4f} [{lo['min']:.4f}, {lo['max']:.4f}], "
              f"upper {hi['mean']:.4f} +- {hi['std']:.4f} [{hi['min']:.4f}, {hi['max']:.4f}]; "
              f"contains the error-free interval in {level['fraction_containing_exact']:.1%} of {level['num_trials']} trials; "
              f"margin {level['margin']:.4f}, within it: {level.get('fraction_within_margin', float('nan')):.1%}")
    if not args.no_write:
        out = jsonio.write_json(OUT_DIR / f"{args.benchmark}.json", record)
        name = FIGURE_NAMES.get(args.benchmark, f"bound_noise_{args.benchmark}")
        fig = figure(record)
        fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
        fig.savefig(OUT_DIR / f"{name}.pdf")
        plt.close(fig)
        print(f"wrote {out}, {OUT_DIR / name}.png and .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
