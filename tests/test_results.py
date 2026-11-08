import json
import math

import jsonschema
import pytest

from tools.common import jsonio, load_benchmark, make_result
from tools.common.results import VERDICTS, agrees_with_expected


@pytest.fixture(scope="module")
def bm():
    return load_benchmark("DUFF_FIN_UNR_BOX")


def test_result_record_is_valid_json_with_encoded_infinities(bm):
    rec = make_result(bm, "krtb", "setting_0", "unknown", {"total": 1.0}, {"T": 50},
                      reach_time_bounds=[[1.0, math.inf]])
    text = jsonio.dumps(rec)
    assert "Infinity" not in text
    back = json.loads(text)
    assert back["reach_time_bounds"] == [[1.0, "inf"]]
    assert jsonio.from_jsonable_float(back["reach_time_bounds"][0][1]) == math.inf
    assert back["agrees_with_expected"] is None


def test_unknown_verdict_is_rejected(bm):
    with pytest.raises(ValueError):
        make_result(bm, "krtb", "setting_0", "reachable", {"total": 1.0}, {})


def test_unknown_tool_is_rejected(bm):
    with pytest.raises(jsonschema.ValidationError):
        make_result(bm, "mytool", "setting_0", "unknown", {"total": 1.0}, {})


@pytest.mark.parametrize(
    "verdict, expected, agrees",
    [
        ("unreachable_proven", "unreachable", True),
        ("unreachable_estimated", "reachable", False),
        ("reachable_witness", "reachable", True),
        ("reachable_possible", "unreachable", None),  # over-approximation proves nothing
        ("unknown", "reachable", None),
        ("error", "unreachable", None),
    ],
)
def test_agreement(verdict, expected, agrees):
    assert verdict in VERDICTS
    assert agrees_with_expected(verdict, expected) is agrees
