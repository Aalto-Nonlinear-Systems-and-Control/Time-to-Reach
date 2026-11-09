"""The generated dReach models encode exactly the benchmark configs.

The hand-written models had a domain excluding the initial set, a dropped
second target box and a 6 s invariant on a 20 s benchmark; these tests
translate the generated formulas back to functions and compare them with the
config's sets and vector field at random points.
"""

import re

import numpy as np
import pytest

from tools.common import expr, list_benchmarks, load_benchmark
from tools.dreach.gen_drh import generate

OPS = ["<=", ">=", "="]


def _split_top(text):
    """Split 'A B C' where each item is a balanced (...) group."""
    items, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                items.append(text[start:i + 1])
    return items


def formula(text, n):
    """Translate a dReach formula into a vectorized predicate X -> bool array."""
    text = text.strip()
    inner = text[1:-1].strip()
    for kw, combine in (("and ", np.logical_and.reduce), ("or ", np.logical_or.reduce)):
        if inner.startswith(kw):
            parts = [formula(p, n) for p in _split_top(inner[len(kw):])]
            return lambda X, parts=parts, combine=combine: combine([p(X) for p in parts])
    depth = 0
    for i, ch in enumerate(inner):  # top-level comparison operator
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            for op in OPS:
                if inner.startswith(f" {op} ", i):
                    lhs = expr.scalar_function(inner[:i], n)
                    rhs = expr.scalar_function(inner[i + len(op) + 2:], n)
                    tol = 1e-12
                    if op == "<=":
                        return lambda X: lhs(X) <= rhs(X) + tol
                    if op == ">=":
                        return lambda X: lhs(X) >= rhs(X) - tol
                    return lambda X: np.abs(lhs(X) - rhs(X)) <= tol
    raise ValueError(f"cannot parse {text!r}")


def section(model, name):
    return re.search(rf"^{name}:\n@1 \(and (.*) \(tau [^\n]*;$", model, re.M).group(1)


@pytest.mark.parametrize("bid", list_benchmarks())
def test_generated_model_matches_config(bid):
    bm = load_benchmark(bid)
    model = generate(bid)
    n = bm.dim
    rng = np.random.default_rng(0)

    # sets: compare membership on points around the sets and in the domain
    for name, s in (("init", bm.initial_set), ("goal", bm.target_set)):
        predicate = formula(section(model, name).rsplit(" (tau", 1)[0], n)
        X = np.vstack([bm.domain.sample(2000, rng), s.sample(500, rng), s.boundary_sample(200, rng)])
        X = np.vstack([X, X[-200:] + 1e-6 * rng.standard_normal(X[-200:].shape)])
        inside, claimed = s.contains(X), predicate(X)
        far = np.abs(s.inside_function(X)) > 1e-9  # ignore points on the boundary
        assert np.array_equal(inside[far], claimed[far]), name

    # flow: -f for backward benchmarks
    flows = re.findall(r"d/dt\[x(\d+)\] = (.*);", model)
    assert [int(i) for i, _ in flows] == list(range(1, n + 1))
    X = bm.initial_set.sample(50, rng)
    F = np.column_stack([expr.scalar_function(e, n)(X) for _, e in flows])
    np.testing.assert_allclose(F, bm.problem_system.rhs(X), rtol=1e-12, atol=1e-12)

    # domain bounds, clock and horizon
    t0, t1 = bm.time_horizon
    for i, (lo, hi) in enumerate(bm.domain.bounds, start=1):
        assert re.search(rf"^\[{lo:g}, {hi:g}\] x{i};$", model, re.M)
    assert f"(tau <= {t1:g});" in model and f"[0, {t1:g}] time;" in model
    assert re.search(r"^jump:\n\n\}", model, re.M)  # no jumps: run with -k 0
