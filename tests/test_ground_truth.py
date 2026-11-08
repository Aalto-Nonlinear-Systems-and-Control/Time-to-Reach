"""Every benchmark's expected_result is consistent with simulation (reduced sample counts).

The full check with more samples is `python -m scripts.check_benchmarks`.
"""

import pytest

from scripts.check_benchmarks import check_benchmark
from tools.common import list_benchmarks


@pytest.mark.parametrize("bid", list_benchmarks())
def test_expected_result_consistent_with_simulation(bid):
    record, passed = check_benchmark(bid, n_interior=60, n_boundary=60, seed=1, jobs=1)
    assert passed, (record["detail"], record["extra"]["problems"])
