import numpy as np
import pytest

from tools.common.sets import Box
from tools.common.simulate import check_trajectory
from tools.common.spec import System


@pytest.mark.parametrize("lo", [0.5, 0.50003, 0.123457])
def test_short_visit_is_detected(lo):
    # x1' = 1 crosses a target of width 1e-5; a 1000-point time grid (what
    # util_global used) cannot see it, and neither can sampling alone when the
    # target is not aligned with the sample times.
    system = System("nonlinear", 1, equations=["1"])
    target = Box([[lo, lo + 1e-5]])
    r = check_trajectory(system, np.array([0.0]), target, (0.0, 1.0))
    assert r.hit
    assert r.first_hit_time == pytest.approx(lo, abs=1e-9)


def test_step_jumping_over_target_is_detected():
    # Exponential growth crosses [0.9, 1.0] within one large solver step.
    system = System("nonlinear", 1, equations=["x1"])
    r = check_trajectory(system, np.array([0.5]), Box([[0.9, 1.0]]), (0.0, 1.0), n_dense=2)
    assert r.hit and r.first_hit_time == pytest.approx(np.log(1.8), rel=1e-9)


def test_miss_reports_distance_and_domain_violation():
    system = System("nonlinear", 2, equations=["-x1", "-x2"])
    target = Box([[2, 3], [2, 3]])
    domain = Box([[0, 0.5], [0, 0.5]])
    r = check_trajectory(system, np.array([1.0, 0.0]), target, (0.0, 2.0), domain=domain)
    assert not r.hit and np.isnan(r.first_hit_time)
    assert r.min_g == pytest.approx(np.hypot(1.0, 2.0), rel=1e-6)
    assert r.max_domain_violation == pytest.approx(0.5, rel=1e-6)


def test_start_inside_target_hits_at_t0():
    system = System("nonlinear", 1, equations=["-x1"])
    r = check_trajectory(system, np.array([0.1]), Box([[0, 1]]), (0.0, 1.0))
    assert r.hit and r.first_hit_time == 0.0


def test_backward_direction_follows_time_reversed_flow():
    # x1' = -x1 decays forward; backward (x1' = +x1) from 0.5 reaches [0.9, 1] at t = ln(1.8).
    forward = System("nonlinear", 1, equations=["-x1"])
    target = Box([[0.9, 1.0]])
    assert not check_trajectory(forward, np.array([0.5]), target, (0.0, 1.0)).hit
    r = check_trajectory(forward.time_reversed(), np.array([0.5]), target, (0.0, 1.0))
    assert r.hit and r.first_hit_time == pytest.approx(np.log(1.8), rel=1e-6)


def test_time_reversal_returns_to_start():
    from scipy.integrate import solve_ivp

    from tools.common import load_benchmark

    bm = load_benchmark("NL_EIG_FIN_BACKWARD_REA_BOX")
    x0 = bm.initial_set.sample(1, np.random.default_rng(0))[0]
    kw = dict(method="DOP853", rtol=1e-11, atol=1e-13)
    y = solve_ivp(bm.problem_system.ode, (0, 0.8), x0, **kw).y[:, -1]
    back = solve_ivp(bm.system.ode, (0, 0.8), y, **kw).y[:, -1]
    np.testing.assert_allclose(back, x0, atol=1e-8)


def test_in_target_intervals_are_exact():
    from tools.common.sets import Union

    system = System("nonlinear", 1, equations=["1"])
    target = Union([Box([[0.2, 0.3]]), Box([[0.6, 0.65]]), Box([[0.80003, 0.80004]])])
    r = check_trajectory(system, np.array([0.0]), target, (0.0, 1.0))
    np.testing.assert_allclose(r.in_target, [(0.2, 0.3), (0.6, 0.65), (0.80003, 0.80004)], atol=1e-9)
    assert r.first_hit_time == pytest.approx(0.2, abs=1e-9)


def test_trajectory_ending_inside_target():
    system = System("nonlinear", 1, equations=["1"])
    r = check_trajectory(system, np.array([0.0]), Box([[0.7, 2.0]]), (0.0, 1.0))
    np.testing.assert_allclose(r.in_target, [(0.7, 1.0)], atol=1e-9)
