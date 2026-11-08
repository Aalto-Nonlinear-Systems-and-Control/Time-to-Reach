import pytest

from scripts.evaluate_results import evaluate, uncovered

INF = float("inf")


@pytest.mark.parametrize(
    "sim, bounds, expected",
    [
        ([(5.6, 5.64)], [(5.4, 5.66)], []),
        ([(5.6, 5.64)], [(5.4, 5.62)], [(5.62, 5.64)]),
        ([(1.0, 2.0)], [(0.0, 1.2), (1.5, INF)], [(1.2, 1.5)]),
        ([(1.0, 1.0)], [(0.0, 0.5)], [(1.0, 1.0)]),  # an instantaneous visit
        ([(1.0, 1.0)], [(0.0, 2.0)], []),
        ([(1.0, 2.0)], [], [(1.0, 2.0)]),
    ],
)
def test_uncovered(sim, bounds, expected):
    assert uncovered(sim, bounds) == pytest.approx(expected)


def _record(verdict, bounds=None):
    return {"benchmark": "B", "verdict": verdict, "expected_result": "reachable",
            "runtime_s": {"total": 1.0}, "reach_time_bounds": bounds}


REACHED = {"verdict": "reachable_witness", "extra": {"in_target_times": [[5.6, 5.64]]}}
MISSED = {"verdict": "unreachable_estimated", "extra": {}}


@pytest.mark.parametrize(
    "verdict, truth, check",
    [
        ("unreachable_estimated", REACHED, "contradicted"),
        ("unreachable_proven", MISSED, "consistent"),
        ("reachable_witness", REACHED, "consistent"),
        ("reachable_estimated", MISSED, "unconfirmed"),
        ("reachable_possible", REACHED, "no claim"),
        ("unsupported", REACHED, "no claim"),
    ],
)
def test_verdict_check(verdict, truth, check):
    assert evaluate(_record(verdict), truth)["verdict_check"] == check


def test_bounds_check():
    row = evaluate(_record("reachable_witness", [[5.41, 5.627]]), REACHED)
    assert row["bounds_cover_simulation"] is False
    assert row["missed_times"] == pytest.approx([(5.627, 5.64)])


def test_no_bounds_check_without_simulated_hits():
    assert "bounds_cover_simulation" not in evaluate(_record("unreachable_estimated", []), MISSED)
