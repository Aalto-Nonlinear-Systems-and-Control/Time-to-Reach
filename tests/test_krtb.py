import numpy as np
import pytest

from tools.common import load_benchmark
from tools.common.settings import load_settings
from tools.common.spec import System
from tools.krtb import analytic, compute_reach_time_bounds, path_integral
from tools.krtb.bounds import periodic_windows

INF = float("inf")


def grid_in(box, k):
    (l1, u1), (l2, u2) = box.bounds
    a, b = np.meshgrid(np.linspace(l1, u1, k), np.linspace(l2, u2, k))
    return np.stack([a.ravel(), b.ravel()], axis=1)


def nl_eig(bid):
    bm = load_benchmark(bid)
    spec = load_settings("krtb", "setting_0", bid)["eigenfunctions"]
    eig = analytic(spec["functions"], spec["eigenvalues"], bm.system)
    return bm, eig


def test_periodic_windows_terminate_without_upper_bound():
    # Regression: an unbounded magnitude interval made the original loop run forever.
    w = periodic_windows((1.0, INF), (0.2, 0.4), period=1.0, t_max=4.0)
    np.testing.assert_allclose(w, [(1.2, 1.4), (2.2, 2.4), (3.2, 3.4)])


def test_periodic_windows_shift_backwards():
    # The phase window can lie after the magnitude lower bound (negative shift);
    # the original code asserted the shift was non-negative.
    w = periodic_windows((0.5, 2.0), (3.6, 3.9), period=1.0, t_max=10.0)
    np.testing.assert_allclose(w, [(0.6, 0.9), (1.6, 1.9)])


def test_periodic_windows_merge_adjacent():
    w = periodic_windows((0.0, 3.0), (0.0, 1.0), period=1.0, t_max=10.0)
    np.testing.assert_allclose(w, [(0.0, 3.0)])


def test_lower_bound_is_clamped_to_zero():
    # One stable eigenvalue; target magnitudes overlap the initial ones, so t may be 0.
    ef0 = np.array([[1.0], [2.0]])
    efF = np.array([[0.5], [1.5]])
    b = compute_reach_time_bounds(ef0, efF, [-1.0])
    assert b.status == "bounded" and b.intervals[0][0] == 0.0


def test_no_information_is_not_unreachable():
    # Regression: infinite bounds were reported as UNREACHABLE.
    ef = np.array([[1.0], [2.0]])
    b = compute_reach_time_bounds(ef, ef, [0.0])
    assert b.status == "no_information" and not b.empty


def test_horizon_clipping_gives_empty():
    ef0 = np.array([[1.0]])
    efF = np.array([[np.exp(-5.0)]])  # reached at t = 5 for lam = -1
    assert compute_reach_time_bounds(ef0, efF, [-1.0], horizon=(0, 10)).intervals == [(5.0, 5.0)]
    assert compute_reach_time_bounds(ef0, efF, [-1.0], horizon=(0, 4)).status == "empty"


def test_backward_equals_equivalent_forward_problem():
    # With both weight cones, a backward problem (X0, XF, -lam) and the equivalent
    # forward problem (XF, X0, lam) give the same bounds. The original LFP used
    # one cone only for mixed-sign spectra and gave [0.64, 1.30] vs [0.76, 0.88].
    bm, eig = nl_eig("NL_EIG_FIN_BACKWARD_REA_BOX")
    e0, eF = eig.evaluate(grid_in(bm.initial_set, 200)), eig.evaluate(grid_in(bm.target_set, 200))
    backward = compute_reach_time_bounds(e0, eF, -eig.eigenvalues)
    forward = compute_reach_time_bounds(eF, e0, eig.eigenvalues)
    np.testing.assert_allclose(backward.intervals, forward.intervals, rtol=1e-6)
    lo, hi = backward.intervals[0]
    assert lo == pytest.approx(0.7605, abs=2e-3) and hi == pytest.approx(0.8835, abs=2e-3)
    # contains the simulated window of first hits [0.7975, 0.8755]
    assert lo <= 0.7975 and hi >= 0.8755


def test_unreachable_backward_benchmark_has_empty_bounds():
    bm, eig = nl_eig("NL_EIG_FIN_BACKWARD_UNR_BOX")
    e0, eF = eig.evaluate(grid_in(bm.initial_set, 100)), eig.evaluate(grid_in(bm.target_set, 100))
    assert compute_reach_time_bounds(e0, eF, -eig.eigenvalues, horizon=bm.time_horizon).status == "empty"


@pytest.mark.parametrize("bid", ["NL_EIG_FIN_BACKWARD_REA_BOX", "NL_SLC_FIN_REA_BOX"])
def test_configured_analytic_eigenfunctions_satisfy_eigen_relation(bid):
    bm = load_benchmark(bid)
    spec = load_settings("krtb", "setting_0", bid)["eigenfunctions"]
    X = bm.initial_set.sample(200, np.random.default_rng(0))
    eig = analytic(spec["functions"], spec["eigenvalues"], bm.system, check_points=X)
    assert eig.info["max_relative_eigen_residual"] < 1e-10


def test_path_integral_matches_known_eigenfunction():
    # x' = -x + x^2 has the eigenfunction x / (1 - x) with eigenvalue -1 at 0.
    eig = path_integral(System("nonlinear", 1, equations=["-x1 + x1^2"]), [0.0], T=30)
    X = np.linspace(-0.5, 0.5, 11)[:, None]
    np.testing.assert_allclose(eig.evaluate(X)[:, 0].real, X[:, 0] / (1 - X[:, 0]), atol=1e-10)
    assert eig.eigenvalues[0] == pytest.approx(-1.0)


def test_path_integral_rejects_non_equilibrium_and_unstable_points():
    with pytest.raises(ValueError):
        path_integral(System("nonlinear", 1, equations=["-x1 + 1"]), [0.0], T=10)
    with pytest.raises(NotImplementedError):
        path_integral(System("nonlinear", 1, equations=["x1"]), [0.0], T=10)


def test_runtime_is_split_into_eigenpairs_and_bounds():
    from tools.krtb.run import run
    rt = run("NL_EIG_FIN_BACKWARD_UNR_BOX")["runtime_s"]
    assert {"setup", "eigenpairs", "bounds", "witness_search"} <= rt.keys()
    assert rt["krtb"] == pytest.approx(rt["eigenpairs"] + rt["bounds"])


def test_samples_add_boundary_and_vertex_points_on_request():
    from tools.common.sets import Box
    from tools.krtb.run import sample
    b = Box([[0, 1], [0, 1]])
    g = np.random.default_rng(0)
    assert len(sample(b, 10, 0, g)) == 10
    X = sample(b, 10, 0, g, n_vertex=5)
    assert len(X) == 15 and np.all(np.isin(X[-5:], (0, 1)))


def test_the_setting_samples_boundaries_and_in_high_dimension_vertices():
    from tools.common.settings import setting_names
    assert setting_names("krtb") == ["setting_0"]  # one setting: the one whose bounds agree with the simulation
    for bid in ("DUFF_FIN_REA_BOX", "NL_SLC_FIN_REA_BOX", "NL_EIG_FIN_BACKWARD_REA_BOX"):
        cfg = load_settings("krtb", "setting_0", bid)
        assert cfg["boundary_samples_initial"] > 0 and cfg["boundary_samples_target"] > 0
        assert "vertex_samples_target" not in cfg
    assert load_settings("krtb", "setting_0", "MAS_CON_FIN_REA_BOX")["vertex_samples_target"] > 0
    assert load_settings("krtb", "setting_0", "CP_LQR_FIN_UNR_SAFE")["vertex_samples_target"] > 0



def test_equilibrium_sample_is_a_target_point_next_to_the_equilibrium():
    from tools.common.sets import Box
    from tools.krtb.run import EQUILIBRIUM_OFFSET, equilibrium_sample
    box = Box([[-0.1, 0.1], [-0.1, 0.1]])
    x = equilibrium_sample(box, [0, 0])
    assert x is not None and box.contains(x[None])[0] and 0 < np.abs(x).max() <= EQUILIBRIUM_OFFSET
    assert equilibrium_sample(Box([[0.5, 1], [0.5, 1]]), [0, 0]) is None
    assert equilibrium_sample(box, None) is None


def test_equilibrium_sample_lifts_the_upper_end_to_the_horizon():
    # x' = diag(-1, -2) x has psi = (x1, x2); from (1, 1) the state enters the box at ln 10 and stays in it,
    # so the reach-time bound has no upper end. Uniform samples miss psi = 0 and cut the bound short.
    from tools.common.sets import Box
    from tools.krtb.run import equilibrium_sample, sample
    box = Box([[-0.1, 0.1], [-0.1, 0.1]])
    lams, e0 = np.array([-1.0, -2.0]), np.array([[1.0, 1.0]])
    XF = sample(box, 1000, 1000, np.random.default_rng(0))
    without = compute_reach_time_bounds(e0, XF, lams, horizon=(0.0, 20.0))
    assert max(hi for _, hi in without.intervals) < 10
    XF = np.vstack([XF, equilibrium_sample(box, [0, 0])])
    with_point = compute_reach_time_bounds(e0, XF, lams, horizon=(0.0, 20.0))
    assert len(with_point.intervals) == 1
    lo, hi = with_point.intervals[0]
    assert lo <= np.log(10) and hi == pytest.approx(20.0)
