import numpy as np
import pytest

from scripts import robustness as rb


def stable_pair(n=300, seed=0, spread=0.1):
    """Values of a complex-conjugate pair of eigenfunctions on two sets, and its eigenvalues."""
    rng = np.random.default_rng(seed)
    z0 = np.exp(rng.uniform(1 - spread, 1 + spread, n) + 1j * rng.uniform(0.0, 0.3, n))
    zF = np.exp(rng.uniform(-1 - spread, -1 + spread, n) + 1j * rng.uniform(1.0, 1.3, n))
    return np.stack([z0, z0.conj()], axis=1), np.stack([zF, zF.conj()], axis=1), np.array([-0.5 + 2j, -0.5 - 2j])


def test_perturbation_is_one_real_factor_per_sample_point():
    rng = np.random.default_rng(0)
    ef = rng.normal(size=(50, 3)) + 1j * rng.normal(size=(50, 3))
    ratio = rb.perturb(ef, 0.01, np.random.default_rng(1)) / ef
    assert np.allclose(ratio.imag, 0)  # the phases are not changed
    assert np.allclose(ratio.real, ratio.real[:, :1])  # the same factor for every eigenfunction
    assert ratio.real.min() >= 0.99 - 1e-12 and ratio.real.max() <= 1.01 + 1e-12
    assert np.ptp(ratio.real[:, 0]) > 0.01  # and it differs from point to point
    assert np.array_equal(rb.perturb(ef, 0.0, rng), ef)


def test_error_margin():
    lams = np.array([-0.25 + 1.39j, -0.25 - 1.39j])
    assert rb.error_margin(0.01, lams) == pytest.approx(np.log(1.01 / 0.99) / 0.25)
    assert rb.error_margin(0.01, np.array([-1.0, 2.5])) is None  # real parts of both signs: no such margin


def test_trials_stay_within_the_margin_and_shrink_with_the_noise():
    ef0, efF, lams = stable_pair()
    exact, _ = rb.magnitude_bound(ef0, efF, lams)
    shift = {}
    for eps in (0.01, 0.001):
        bounds = rb.run_trials(ef0, efF, lams, eps, 20, np.random.default_rng([0, 0]))
        stats = rb.statistics(bounds, exact, rb.error_margin(eps, lams))
        assert stats["num_trials"] == 20 and stats["num_empty"] == 0
        assert stats["fraction_within_margin"] == 1.0
        shift[eps] = max(stats["max_abs_shift"].values())
    assert shift[0.001] < shift[0.01]


def test_trials_are_reproducible():
    ef0, efF, lams = stable_pair()
    a = rb.run_trials(ef0, efF, lams, 0.01, 5, np.random.default_rng([0, 0]))
    b = rb.run_trials(ef0, efF, lams, 0.01, 5, np.random.default_rng([0, 0]))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, rb.run_trials(ef0, efF, lams, 0.01, 5, np.random.default_rng([0, 1])))


def test_no_noise_gives_the_error_free_interval():
    ef0, efF, lams = stable_pair()
    exact, _ = rb.magnitude_bound(ef0, efF, lams)
    bounds = rb.run_trials(ef0, efF, lams, 0.0, 2, np.random.default_rng(0))
    assert np.allclose(bounds, exact)


def test_defaults_reproduce_the_original_experiment_outside_the_benchmark_results():
    from tools.common import paths

    assert rb.DEFAULT_BENCHMARK == "DUFF_FIN_REA_NONCONVEX_LEVEL_SET"
    assert rb.DEFAULT_SAMPLES == 1000  # uniform points in each set, as in the original experiment
    assert rb.DEFAULT_SETTING == "setting_0"  # the error-free reference is the KRTB data of the comparison
    assert rb.OUT_DIR == paths.RESULTS_DIR / "robustness"
    assert paths.SUMMARY_DIR not in rb.OUT_DIR.parents and rb.OUT_DIR != paths.SUMMARY_DIR


def test_extremal_points_give_the_samples_the_interval_of_the_reference_data():
    ef0, efF, lams = stable_pair()
    ref0, refF, _ = stable_pair(n=2000, seed=1, spread=0.15)  # reference data that reach further than the samples
    reference, _ = rb.magnitude_bound(ref0, refF, lams)
    own, _ = rb.magnitude_bound(ef0, efF, lams)
    assert not np.allclose(own, reference)
    assert len(rb.extremal_rows(ref0)) == 2  # a conjugate pair shares its extremal points
    joined, _ = rb.magnitude_bound(np.vstack([ef0, rb.extremal_rows(ref0)]), np.vstack([efF, rb.extremal_rows(refF)]), lams)
    assert np.allclose(joined, reference)


def test_experiment_record_and_figure():
    # analytic eigenfunctions, so this is fast
    record = rb.experiment("NL_SLC_FIN_REA_BOX", "setting_0", 200, eps_levels=(0.01,), trials=3, seed=0)
    assert record["matches_stored_result"] in (True, None)  # None: no stored KRTB result
    assert record["error_free"][0] < record["error_free"][1]
    assert record["num_added_extremal_points"] == {"initial": 2, "target": 2}
    assert record["num_samples"] == {"initial": 202, "target": 202}
    same = rb.experiment("NL_SLC_FIN_REA_BOX", "setting_0", None, eps_levels=(0.01,), trials=1, seed=0)
    assert same["num_added_extremal_points"] == {"initial": 0, "target": 0} and same["error_free"] == record["error_free"]
    (level,) = record["levels"]
    assert level["eps"] == 0.01 and level["num_trials"] == 3
    assert len(level["lower_bounds"]) == len(level["upper_bounds"]) == 3
    fig = rb.figure(record)
    assert len(fig.axes) == 2
    rb.plt.close(fig)
