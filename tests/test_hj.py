import jax.numpy as jnp
import numpy as np
import pytest

from tools.common import jsonio, load_benchmark
from tools.hj.model import implicit_function, vector_field
from tools.hj.run import analyze


@pytest.mark.parametrize("bid", ["DUFF_FIN_REA_DISJOINT_BOX", "DUFF_FIN_REA_BALL", "DUFF_FIN_REA_BOX",
                                 "DUFF_FIN_REA_NONCONVEX_LEVEL_SET", "CP_LQR_FIN_REA_UNSAFE"])
def test_implicit_functions_match_common_sets(bid):
    bm = load_benchmark(bid)
    g = implicit_function(bm.raw["target_sets"], bm.dim)
    X = np.vstack([bm.domain.sample(300, np.random.default_rng(0)), bm.target_set.sample(100, np.random.default_rng(1))])
    ours = np.array([float(g(jnp.asarray(x, dtype=jnp.float32))) for x in X])
    common = bm.target_set.inside_function(X)
    if all(s["type"] != "level_set" for s in bm.raw["target_sets"]):
        np.testing.assert_allclose(ours, common, rtol=1e-4, atol=1e-5)  # the same signed distance
    else:  # the common sets clip the inside of a level set at 0; HJ keeps the function: same membership
        assert np.array_equal(ours <= 1e-6, common <= 0)
        assert np.all(ours[common < 0] < 0)


def test_level_set_target_values_are_the_function_itself():
    import hj_reachability as hj

    from tools.hj.model import target_values

    # the disc x1^2 + x2^2 <= 0.25 written with a scaled function: the values on the grid are that function
    spec = [{"type": "level_set", "function": "100*(x1^2 + x2^2 - 0.25)", "bounding_box": [[-1, 1], [-1, 1]]}]
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])), (101, 101))
    v = np.asarray(target_values(spec, 2, grid))
    X = np.asarray(grid.states)
    expected = np.maximum(100 * (np.sum(X ** 2, axis=-1) - 0.25), np.max(np.abs(X) - 1, axis=-1))  # f, within the box
    np.testing.assert_allclose(v, expected, rtol=1e-4, atol=1e-3)  # no re-distancing, no tolerance
    r = np.linalg.norm(X, axis=-1)
    away = np.abs(r - 0.5) > 1e-3  # nodes on the circle itself round either way in float32
    assert np.array_equal((v <= 0)[away], (r <= 0.5)[away])


def test_a_grid_larger_than_the_memory_limit_is_not_started():
    from tools.hj.run import grid_memory, run

    nodes, least = grid_memory((100,) * 16)
    assert nodes == 100 ** 16 and least == 100 ** 16 * 4 * 17  # exact integers
    assert grid_memory((4,) * 16)[1] > 8e9 > grid_memory((3,) * 16)[1]  # 16D: 4 nodes per dimension are too many
    record = run("MAS_CON_FIN_REA_BOX")
    assert record["verdict"] == "out_of_memory" and "not started" in record["detail"]
    assert record["extra"]["started"] is False and record["extra"]["memory_limit_gb"] == 8


def test_vector_field_is_the_one_of_the_benchmark():
    bm = load_benchmark("NL_SLC_FIN_REA_BOX")
    F = vector_field(bm.system)
    for x in (np.array([1.3, 1.35]), np.array([0.1, 0.0])):  # also near the singular origin: nothing is cut out
        np.testing.assert_allclose(np.asarray(F(jnp.asarray(x, dtype=jnp.float32))), bm.system.rhs(x[None])[0], rtol=1e-5)


def toy(tmp_path, target, direction="forward"):
    """x' = -x in 2D from the box [0.8, 1] x [-0.1, 0.1]; trajectories move toward the origin."""
    cfg = {
        "id": "TOY", "description": "toy", "direction": direction,
        "system": {"type": "nonlinear", "equations": ["-x1", "-x2"]},
        "domain": [[-0.5, 1.5], [-1, 1]],
        "initial_sets": [{"type": "box", "bounds": [[0.8, 1.0], [-0.1, 0.1]]}],
        "target_sets": [target],
        "time_horizon": [0, 1.0],
        "expected_result": "reachable",
    }
    path = tmp_path / "TOY.json"
    jsonio.write_json(path, cfg)
    return load_benchmark(path)


SETTINGS = {"grid_size": 101, "accuracy": "medium", "time_step_report": 0.02,
            "num_samples_initial": 200, "boundary_samples_initial": 200, "seed": 0}


@pytest.mark.parametrize(
    "target, direction, verdict",
    [
        ({"type": "box", "bounds": [[0.2, 0.5], [-0.2, 0.2]]}, "forward", "reachable_estimated"),  # x1(1) ~ 0.33
        ({"type": "box", "bounds": [[-0.4, -0.2], [-0.2, 0.2]]}, "forward", "unreachable_estimated"),
        ({"type": "box", "bounds": [[1.3, 1.45], [-0.2, 0.2]]}, "backward", "reachable_estimated"),  # x1 grows
        ({"type": "box", "bounds": [[0.2, 0.5], [-0.2, 0.2]]}, "backward", "unreachable_estimated"),
    ],
)
def test_verdicts_flip_with_the_problem(tmp_path, target, direction, verdict):
    # Smoke test of the whole pipeline: the verdict must depend on the problem.
    got, detail, _, extra = analyze(toy(tmp_path, target, direction), SETTINGS)
    assert got == verdict, detail
    if verdict == "reachable_estimated" and direction == "forward":
        # first entry of x1 = exp(-t) into x1 <= 0.5 from x1(0) = 0.8: t = ln(1.6) ~ 0.47
        assert extra["first_reach_time_estimate"] == pytest.approx(np.log(1.6), abs=0.06)
