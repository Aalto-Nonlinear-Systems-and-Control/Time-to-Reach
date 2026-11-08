"""Linear-fractional programs over eigenfunction weights.

Bounds on the reach time t come from weighted combinations of the per-
eigenfunction relations  Re(lam_i) * t = Delta_i  (or Im(lam_i) * t = Delta_i).
With weights y in [0, 1]^m this gives t = c^T y / d^T y. Dividing by d^T y
flips inequalities when d^T y < 0, so every program here is solved on the cone
{d^T y > 0}; callers that need the cone {d^T y < 0} pass (-c, -d), which leaves
the ratio unchanged. The Charnes-Cooper substitution z = s*y, d^T z = 1 turns the
ratio into the linear objective c^T z.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

# cvxpy statuses whose value is meaningful: optimal values, +-inf for unbounded
# programs (a valid bound: no finite t is consistent), and -inf/+inf for
# infeasible maximisation/minimisation (the cone is empty: no information).
_USABLE = {
    cp.OPTIMAL,
    cp.OPTIMAL_INACCURATE,
    cp.UNBOUNDED,
    cp.UNBOUNDED_INACCURATE,
    cp.INFEASIBLE,
    cp.INFEASIBLE_INACCURATE,
}


@dataclass
class LFPResult:
    status: str
    value: float
    weights: np.ndarray  # y = z / s for optimal solutions, else None


def solve(c, d, maximize, solver=cp.MOSEK):
    """Optimise c^T y / d^T y over y in [0, 1]^m restricted to d^T y > 0."""
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    z = cp.Variable(c.shape[0])
    s = cp.Variable(nonneg=True)
    objective = cp.Maximize(c @ z) if maximize else cp.Minimize(c @ z)
    problem = cp.Problem(objective, [d @ z == 1, z >= 0, z <= s])
    problem.solve(solver=solver)
    if problem.status not in _USABLE:
        raise RuntimeError(f"linear-fractional program failed with status {problem.status!r}")
    weights = None
    if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and s.value is not None and s.value > 0:
        weights = z.value / s.value
    return LFPResult(problem.status, float(problem.value), weights)
