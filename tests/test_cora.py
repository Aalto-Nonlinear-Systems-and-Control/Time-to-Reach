"""Smoke tests for the CORA pipeline (need MATLAB; run with `-m matlab`)."""

import pytest

from tools.common import jsonio, load_benchmark
from tools.cora.run import execute, interpret, matlab_command

try:
    matlab_command("R2025a")
    HAVE_MATLAB = True
except RuntimeError:
    HAVE_MATLAB = False

pytestmark = [pytest.mark.matlab, pytest.mark.skipif(not HAVE_MATLAB, reason="MATLAB not found")]

SETTINGS = {"matlab_release": "R2025a", "alg": "lin", "tensorOrder": 2, "taylorTerms": 4, "zonotopeOrder": 50,
            "timeStep": 0.01, "timeout_s": 900}


def toy(tmp_path, target):
    """x' = -x in 2D from [0.8, 1] x [-0.1, 0.1]; x1(t) = x1(0) exp(-t)."""
    cfg = {
        "id": "TOY", "description": "toy", "direction": "forward",
        "system": {"type": "nonlinear", "equations": ["-x1", "-x2 + 0*x1^2"]},
        "domain": [[-0.5, 1.5], [-1, 1]],
        "initial_sets": [{"type": "box", "bounds": [[0.8, 1.0], [-0.1, 0.1]]}],
        "target_sets": [target],
        "time_horizon": [0, 1.0],
        "expected_result": "reachable",
    }
    path = tmp_path / "TOY.json"
    jsonio.write_json(path, cfg)
    return load_benchmark(path)


@pytest.mark.parametrize(
    "bounds, verdict",
    [
        ([[0.0, 0.6], [-0.5, 0.5]], "reachable_proven"),  # x1 <= 0.6 for t >= ln(1/0.6): whole sets inside
        # x1(1) is in [0.294, 0.368]: the sets only overlap x1 <= 0.3, which proves nothing
        ([[0.0, 0.3], [-0.5, 0.5]], "reachable_possible"),
        ([[-0.4, -0.2], [-0.2, 0.2]], "unreachable_proven"),
    ],
)
def test_verdicts_flip_with_the_problem(tmp_path, bounds, verdict):
    out = execute(toy(tmp_path, {"type": "box", "bounds": bounds}), SETTINGS, tmp_path / "work")
    got, detail = interpret(out)
    assert got == verdict, detail
    assert (out["num_inside"] > 0) == (verdict == "reachable_proven")
    assert (out["num_possible"] + out["num_inside"] == 0) == (verdict == "unreachable_proven")
