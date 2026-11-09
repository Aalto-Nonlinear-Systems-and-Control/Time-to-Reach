"""The figures of the paper's examples (scripts.paper_figures) are drawn from the stored results."""

import numpy as np

from scripts import paper_figures as pf


def test_interval_formatting():
    assert pf.fmt((5.048, 5.8307)) == "[5.05, 5.83]"
    assert pf.fmt((8.5504, 12.0), 3) == "[8.550, 12]"  # the horizon end without decimals
    assert pf.ceil2(5.9626) == 5.97 and pf.ceil2(5.84) == 5.84


def test_duffing_figure_uses_the_stored_bounds():
    fig, data = pf.figure_duffing("DUFF_FIN_REA_DISJOINT_BOX", 30, np.random.default_rng(0))
    assert len(fig.axes) == 1 and len(data["reach_time_bounds"]) == 2
    assert data["horizon"][1] == pf.ceil2(data["reach_time_bounds"][1][1])
    assert len(data["trajectory"]["t"]) == len(data["trajectory"]["x"])
    pf.plt.close(fig)


def test_cartpole_figures():
    fig, data = pf.figure_cartpole_time("CP_LQR_FIN_REA_UNSAFE")
    assert data["collision_time_bounds"][0] == 0.0 and data["reach_time_bounds"][1] == 12.0
    walls = pf.cartpole_walls()
    assert walls == {"X_U^1": 1.3, "X_U^2": 3.7}  # read from the targets of the configs
    assert max(data["trajectory"]["pole_position"]) > walls["X_U^1"]  # the pole reaches the first wall ...
    pf.plt.close(fig)
    fig, data = pf.figure_cartpole_reach("CP_LQR_FIN_REA_UNSAFE")
    assert data["wall"] == 3.7 and data["max_pole_position_in_interval"] < 3.7  # ... but never the second
    assert data["broken_axis"]  # the far wall is drawn behind a break of the p axis
    pf.plt.close(fig)
    fig, data = pf.figure_cartpole_collision("CP_LQR_FIN_REA_UNSAFE")
    assert data["wall"] == 1.3 and not data["broken_axis"]
    np.testing.assert_allclose([data["snapshot_times"][0], data["snapshot_times"][-1]], data["interval"], atol=1e-4)
    pf.plt.close(fig)
