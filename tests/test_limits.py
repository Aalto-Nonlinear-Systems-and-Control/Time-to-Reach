"""Memory and time limits of a computation in a child process (tools.common.limits)."""

import time

import numpy as np
import pytest

from tools.common.limits import GB, ChildFailed, MemoryLimitExceeded, TimeLimitExceeded, footprint, run_limited


def allocate(gb, hold_s=0.0):
    """Touch `gb` GB in blocks of 0.1 GB, then return the footprint."""
    blocks = [np.ones(int(0.1 * GB / 8)) for _ in range(round(gb / 0.1))]
    time.sleep(hold_s)
    return len(blocks), footprint()


def fail():
    raise ValueError("no such thing")


def refuse():
    raise MemoryError("Unable to allocate 1.0 EiB")


def test_result_and_peak_memory_are_returned():
    (blocks, inside), peak = run_limited(allocate, (0.3, 0.3), memory_limit=2 * GB, timeout=60)  # held for 0.3 s
    assert blocks == 3 and peak >= 0.3 * GB and inside <= peak + 0.1 * GB


def test_stops_at_the_memory_limit():
    t0 = time.perf_counter()
    with pytest.raises(MemoryLimitExceeded) as info:
        run_limited(allocate, (3.0, 30.0), memory_limit=1 * GB, timeout=60)
    assert time.perf_counter() - t0 < 20  # stopped while allocating, not after the 30 s hold
    assert 1 * GB < info.value.used < 2 * GB and info.value.limit == 1 * GB  # stopped soon after the limit


def test_a_refused_allocation_counts_as_out_of_memory():
    with pytest.raises(MemoryLimitExceeded) as info:
        run_limited(refuse, memory_limit=1 * GB)
    assert "Unable to allocate" in info.value.note


def test_time_limit_and_errors():
    with pytest.raises(TimeLimitExceeded):
        run_limited(allocate, (0.1, 30.0), memory_limit=1 * GB, timeout=1.0)
    with pytest.raises(ChildFailed, match="no such thing"):
        run_limited(fail)
