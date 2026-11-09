import copy

import jsonschema
import numpy as np
import pytest

from tools.common import jsonio, list_benchmarks, load_benchmark
from tools.common.spec import benchmark_path

BENCHMARKS = list_benchmarks()


def test_benchmarks_exist():
    assert len(BENCHMARKS) == 16


@pytest.mark.parametrize("bid", BENCHMARKS)
def test_config_loads(bid):
    bm = load_benchmark(bid)
    assert bm.id == bid
    X = bm.initial_set.sample(5, np.random.default_rng(0))
    F = bm.system.rhs(X)
    assert F.shape == (5, bm.dim) and np.all(np.isfinite(F))


@pytest.mark.parametrize("bid", BENCHMARKS)
def test_expected_result_matches_name(bid):
    # Regression: MAS_CON_FIN_UNR_SAFE_* and CP_LQR UNR were labelled 'reachable'.
    expected = load_benchmark(bid).expected_result
    if "_REA_" in bid:
        assert expected == "reachable"
    elif "_UNR_" in bid:
        assert expected == "unreachable"
    else:
        pytest.fail(f"{bid}: name says neither REA nor UNR")


@pytest.mark.parametrize("bid", BENCHMARKS)
def test_sets_inside_domain(bid):
    bm = load_benchmark(bid)
    for s in (bm.initial_set, bm.target_set):
        bb = s.bounding_box()
        assert np.all(bb[:, 0] >= bm.domain.lower) and np.all(bb[:, 1] <= bm.domain.upper)


def _base():
    return jsonio.read_json(benchmark_path("DUFF_FIN_REA_BOX"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.pop("expected_result"),
        lambda c: c.update(expected_result="Reachable"),
        lambda c: c["target_sets"].append({"type": "polytope", "A": []}),
        lambda c: c.update(unknown_key=1),
        lambda c: c["initial_sets"][0].update(bounds=[[0, 1]]),  # wrong dimension
        lambda c: c.update(time_horizon=[1, 1]),
        lambda c: c["system"].update(equations=["x2", "x1 + p"]),  # free parameter
        lambda c: c.update(direction="reverse"),
        lambda c: c["system"].update(parameters={"x3": 1.0}),  # name of a state variable
        lambda c: c["system"].update(parameters={"sin": 1.0}),  # name of a function
        lambda c: c["system"].update(parameters={"k": "1"}),  # not a number
        lambda c: c["system"].update(definitions={"u": "-k*x1"}),  # k is not defined
        lambda c: c["system"].update(parameters={"u": 1.0}, definitions={"u": "x1"}),  # defined twice
    ],
)
def test_invalid_configs_are_rejected(tmp_path, mutate):
    cfg = copy.deepcopy(_base())
    mutate(cfg)
    path = tmp_path / "DUFF_FIN_REA_BOX.json"
    jsonio.write_json(path, cfg)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        load_benchmark(path)


@pytest.mark.parametrize("bid", ["CP_LQR_FIN_REA_CONVERGE", "CP_LQR_FIN_REA_UNSAFE", "CP_LQR_FIN_UNR_SAFE"])
def test_cart_pole_configs_are_tedrakes_model_with_the_designed_lqr(bid):
    """The three cart-pole configs carry Tedrake's cart-pole (hanging equilibrium at the origin) with the paper's
    parameters and the LQR gain designed by scripts/cartpole_lqr.py (see its docstring)."""
    from scripts.cartpole_lqr import X0, check_model, linearisation, lqr, DEFAULT_Q, DEFAULT_R
    K, _ = lqr(*linearisation(), DEFAULT_Q, DEFAULT_R)
    check = check_model(K)["configs"][bid]
    assert check["parameters_equal"] and check["equations_equal"] and check["gains_equal"]
    bm = load_benchmark(bid)
    assert "u" in bm.raw["system"]["definitions"]
    np.testing.assert_allclose(bm.initial_set.bounds[:, 0], X0, atol=1e-6)  # the initial set is the point x0
    np.testing.assert_allclose(bm.initial_set.bounds[:, 1], X0, atol=1e-6)
    assert bm.solver_config()["system"] == {"type": "nonlinear", "equations": bm.system.equations}


@pytest.mark.parametrize("bid", BENCHMARKS)
def test_direction_matches_name(bid):
    bm = load_benchmark(bid)
    assert bm.direction == ("backward" if "_BACKWARD_" in bid else "forward")
