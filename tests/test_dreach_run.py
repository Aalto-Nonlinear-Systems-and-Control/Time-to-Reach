"""Smoke tests for the dReach pipeline (need Docker and the dReach image; run with `-m docker`)."""

import shutil
import subprocess

import pytest

from tools.common import jsonio, load_benchmark
from tools.dreach.run import execute, interpret

IMAGE = "dreach:dreal3-7f31797b"
SETTINGS = {"image": IMAGE, "k": 0, "precision": 0.001, "timeout_s": 600}


def _image_available():
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


pytestmark = [pytest.mark.docker,
              pytest.mark.skipif(not _image_available(), reason=f"Docker image {IMAGE} not available")]


def toy(tmp_path, target, direction="forward"):
    """x' = -x in 2D from [0.8, 1] x [-0.1, 0.1]; x1(t) = x1(0) exp(-t)."""
    cfg = {
        "id": "TOY", "description": "toy", "direction": direction,
        "system": {"type": "nonlinear", "equations": ["-x1", "-x2"]},
        "domain": [[-0.5, 2.5], [-1, 1]],
        "initial_sets": [{"type": "box", "bounds": [[0.8, 1.0], [-0.1, 0.1]]}],
        "target_sets": target if isinstance(target, list) else [target],
        "time_horizon": [0, 1.0],
        "expected_result": "reachable",
    }
    path = tmp_path / "TOY.json"
    jsonio.write_json(path, cfg)
    return load_benchmark(path)


REACHABLE = {"type": "box", "bounds": [[0.0, 0.6], [-0.5, 0.5]]}
UNREACHABLE = {"type": "box", "bounds": [[-0.4, -0.2], [-0.2, 0.2]]}


@pytest.mark.parametrize(
    "target, direction, verdict",
    [
        (REACHABLE, "forward", "reachable_possible"),
        (UNREACHABLE, "forward", "unreachable_proven"),
        # backward: x' = +x, so x1 grows from [0.8, 1] past 1.2 before t = ln 1.5 < 1
        ({"type": "box", "bounds": [[1.2, 2.0], [-0.5, 0.5]]}, "backward", "reachable_possible"),
        # a union of targets is reachable if one member is ('or' in the model)
        ([UNREACHABLE, REACHABLE], "forward", "reachable_possible"),
        ({"type": "ball", "center": [-0.3, 0.0], "radius": 0.1}, "forward", "unreachable_proven"),
    ],
)
def test_verdicts_flip_with_the_problem(tmp_path, target, direction, verdict):
    out = execute(toy(tmp_path, target, direction), SETTINGS, tmp_path / "work")
    got, detail = interpret(out, SETTINGS)
    assert got == verdict, detail
    assert out["dreach_time"] is not None, out["stderr"]
