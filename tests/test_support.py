"""What the runners hand to the compared tools: only what a tool handles itself."""

import pytest

from tools.common import list_benchmarks, load_benchmark
from tools.common.support import box_to_box_reason

BOX_TO_BOX = {"DUFF_FIN_REA_BOX", "DUFF_FIN_UNR_BOX", "NL_SLC_FIN_REA_BOX", "NL_SLC_FIN_UNR_BOX",
              "CP_LQR_FIN_REA_CONVERGE", "MAS_CON_FIN_REA_BOX", "MAS_CON_FIN_UNR_SAFE_0", "MAS_CON_FIN_UNR_SAFE_1"}


def test_set_propagation_tools_get_forward_box_to_box_problems():
    assert {bid for bid in list_benchmarks() if box_to_box_reason(load_benchmark(bid)) is None} == BOX_TO_BOX


@pytest.mark.parametrize(
    "bid, why",
    [("DUFF_FIN_REA_BALL", "ball or a level set"), ("DUFF_FIN_REA_NONCONVEX_LEVEL_SET", "ball or a level set"),
     ("CP_LQR_FIN_UNR_SAFE", "ball or a level set"), ("DUFF_FIN_REA_DISJOINT_BOX", "several initial or target sets"),
     ("NL_EIG_FIN_BACKWARD_REA_BOX", "backward")],
)
def test_other_benchmarks_are_unsupported_without_starting_the_tool(bid, why):
    from tools.cora.run import run as cora
    from tools.juliareach.run import run as juliareach

    assert why in box_to_box_reason(load_benchmark(bid))
    for run in (cora, juliareach):  # needs neither MATLAB nor Julia
        record = run(bid)
        assert record["verdict"] == "unsupported" and why in record["detail"]
        assert record["reach_time_bounds"] is None

