import numpy as np

from scripts.cartpole_lqr import (DEFAULT_Q, DEFAULT_R, PARAMETERS, X0, basin_check, check_model, linearisation, lqr,
                                  metrics, simulate)


def test_equations_are_tedrakes_cart_pole_around_the_hanging_equilibrium():
    check = check_model()
    # the equations (Tedrake's, as printed) follow from the Lagrangian (point mass on a rod, theta from the hanging vertical)
    assert check["equations_equal_lagrangian"]
    A, B = linearisation()
    # open loop at the hanging equilibrium: an undamped swing +-j sqrt((m_c + m_p) g / (m_c l)) and the cart's double zero
    w = np.sqrt((PARAMETERS["m_c"] + PARAMETERS["m_p"]) * PARAMETERS["g"] / (PARAMETERS["m_c"] * PARAMETERS["l"]))
    ev = np.linalg.eigvals(A)
    assert np.allclose(ev.real, 0) and np.allclose(sorted(ev.imag), [-w, 0, 0, w])
    assert np.allclose(B.ravel(), [0, 1 / PARAMETERS["m_c"], 0, -1 / (PARAMETERS["m_c"] * PARAMETERS["l"])])


def test_lqr_damps_the_swing_and_the_example_story():
    A, B = linearisation()
    K, eig = lqr(A, B, DEFAULT_Q, DEFAULT_R)
    assert np.allclose(K, [2.907349, 4.876613, -3.770219, -1.639673], atol=1e-6)
    re = np.sort(eig.real)
    assert re[-1] < 0 and re[0] - 2 * re[-1] > 0  # stable, and KRTB's path integral converges (2 max Re < min Re)
    zeta = min(-e.real / abs(e) for e in eig if e.imag > 0)
    assert 0.2 < zeta < 0.25  # a lightly damped swing: the pole swings back and forth a few times
    t, X, u = simulate(K, X0, t_end=12.0)
    m = metrics(t, X, u)
    # the pole tip (the hanging mass) passes the wall at 1.3 inside the target box and never the one at 3.7
    assert m["walls"]["X_U^1"]["hit"] and m["walls"]["X_U^1"]["inside_box"] and not m["walls"]["X_U^2"]["hit"]
    assert 0.8 < m["walls"]["X_U^1"]["intervals"][0][0] < 0.85 and 1.35 < m["tip_range"][1] < 1.37
    assert np.sum(np.diff(np.sign(X[:, 2])) != 0) >= 6  # back and forth
    assert np.abs(X[-1]).max() < 1e-3
    # the target boxes lie in the basin of attraction (a small sample of what --basin checks)
    assert list(basin_check(K, n=40))[-1] == "all samples converge"
