"""SOS barrier certificates: what is supported (no MATLAB) and smoke tests (MATLAB, `-m matlab`)."""

from unittest import mock

import pytest

import tools.sosbc.run as sosbc
from tools.common import jsonio, load_benchmark
from tools.cora.run import matlab_command


@pytest.mark.parametrize(
    "bid, supported",
    [("DUFF_FIN_UNR_BOX", True), ("NL_SLC_FIN_UNR_BOX", False), ("NL_EIG_FIN_BACKWARD_UNR_BOX", False),
     ("CP_LQR_FIN_UNR_SAFE", False), ("MAS_CON_FIN_UNR_SAFE_0", False)],
)
def test_only_polynomial_dynamics_are_supported(bid, supported):
    bm = load_benchmark(bid)
    if supported:
        field, degree = sosbc.polynomial_field(bm)
        assert degree == 3 and len(field) == bm.dim
    else:  # trigonometric or exponential terms; NL_SLC is rational and is not multiplied out
        with pytest.raises(sosbc.Unsupported):
            sosbc.polynomial_field(bm)
        assert sosbc.run(bid)["verdict"] == "unsupported"


def test_only_polynomial_level_sets_are_supported():
    box = [[-1, 1], [-1, 1]]
    assert sosbc.set_spec({"type": "level_set", "function": "x1^2 + x2^2 - 1", "bounding_box": box}, 2)["kind"] == "poly"
    with pytest.raises(sosbc.Unsupported):  # not replaced by its bounding box
        sosbc.set_spec({"type": "level_set", "function": "sin(x1) + x2", "bounding_box": box}, 2)
    record = sosbc.run("DUFF_FIN_REA_NONCONVEX_LEVEL_SET")
    assert record["verdict"] == "unsupported" and "non-polynomial level set" in record["detail"]


try:
    matlab_command("R2025a")
    HAVE_MATLAB = True
except RuntimeError:
    HAVE_MATLAB = False


def toy(tmp_path, target):
    cfg = {
        "id": "TOY", "description": "toy",
        "system": {"type": "nonlinear", "equations": ["-x1 - x1^3", "-x2"]},
        "domain": [[-1.5, 1.5], [-1, 1]],
        "initial_sets": [{"type": "box", "bounds": [[0.8, 1.0], [-0.1, 0.1]]}],
        "target_sets": [target],
        "time_horizon": [0, 1.0],
        "expected_result": "unreachable",
    }
    path = tmp_path / "TOY.json"
    jsonio.write_json(path, cfg)
    return load_benchmark(path)


@pytest.mark.matlab
@pytest.mark.skipif(not HAVE_MATLAB, reason="MATLAB not found")
@pytest.mark.parametrize(
    "target, verdict",
    [
        ({"type": "box", "bounds": [[-1.2, -0.8], [-0.2, 0.2]]}, "unreachable_proven"),  # other side of 0
        ({"type": "box", "bounds": [[0.2, 0.6], [-0.2, 0.2]]}, "unknown"),  # reached: no certificate
    ],
)
def test_certificate_only_for_unreachable_targets(tmp_path, target, verdict):
    bm = toy(tmp_path, target)
    cfg = {"matlab_release": "R2025a", "deg_B": 6, "boundary_condition": True, "timeout_s": 900}
    with mock.patch.object(sosbc, "load_benchmark", lambda _: bm), \
            mock.patch.object(sosbc, "load_settings", lambda *a: dict(cfg)):
        rec = sosbc.run("TOY")
    assert rec["verdict"] == verdict, rec["detail"]
