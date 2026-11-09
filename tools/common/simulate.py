"""Numerical simulation with target-hit detection.

The target's inside function g (g <= 0 exactly on the target) is evaluated on
the solver's dense output at many points inside every integration step, not on
a fixed time grid, and the first entry time is refined by root finding. A plain
step-endpoint event test is not enough: g is positive on both sides of a box, so
a step that passes through the whole target shows no sign change. These are
numerical witnesses, not proofs.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

RTOL = 1e-9
ATOL = 1e-11
POINTS_PER_STEP = 32


@dataclass
class TrajectoryCheck:
    x0: np.ndarray
    hit: bool
    first_hit_time: float  # nan if no hit
    min_g: float  # min over [t0, t1] of the target inside function (<= 0 means inside)
    min_g_time: float
    max_domain_violation: float  # > 0 if the trajectory leaves the domain
    success: bool
    message: str
    in_target: list = field(default_factory=list)  # (t_enter, t_leave) intervals within the horizon


def _sample_times(step_ends, n_dense, points_per_step):
    """Uniform grid plus points_per_step points inside every integration step."""
    frac = np.linspace(0.0, 1.0, points_per_step, endpoint=False)
    starts, lengths = step_ends[:-1], np.diff(step_ends)
    within = (starts[:, None] + lengths[:, None] * frac[None, :]).ravel()
    grid = np.linspace(step_ends[0], step_ends[-1], n_dense)
    return np.unique(np.concatenate([grid, within, step_ends]))


def merge_intervals(intervals, tol=0.0):
    """Union of closed intervals, merging those that overlap or are within tol."""
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1] + tol:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def check_trajectory(system, x0, target, time_horizon, domain=None, n_dense=2001, method="DOP853",
                     points_per_step=POINTS_PER_STEP, max_step=np.inf):
    """Integrate one trajectory over the horizon and report when it is in the target set."""
    t0, t1 = time_horizon
    sol = solve_ivp(
        system.ode, (t0, t1), np.asarray(x0, float), method=method,
        rtol=RTOL, atol=ATOL, dense_output=True, max_step=max_step,
    )
    if sol.status < 0:
        return TrajectoryCheck(np.asarray(x0), False, np.nan, np.nan, np.nan, np.nan, False, sol.message)

    ts = _sample_times(sol.t, n_dense, points_per_step)
    X = sol.sol(ts).T
    gv = target.inside_function(X)
    last = len(ts) - 1

    def g_of_t(t):
        return target.inside_function(sol.sol(t)[None, :])[0]

    def crossing(outside_t, inside_t):
        """Boundary time between a sample outside (g > 0) and one inside (g <= 0)."""
        if g_of_t(inside_t) == 0:
            return float(inside_t)
        lo, hi = sorted((outside_t, inside_t))
        return float(brentq(g_of_t, lo, hi, xtol=1e-12))

    k = int(np.argmin(gv))
    min_g, min_g_time = float(gv[k]), float(ts[k])
    intervals = []

    # Runs of samples inside the target, with refined entry and exit times.
    inside = gv <= 0
    if inside.any():
        edges = np.flatnonzero(np.diff(inside.astype(int)))
        starts = [0] if inside[0] else []
        starts += [e + 1 for e in edges if not inside[e]]
        ends = [e for e in edges if inside[e]] + ([last] if inside[last] else [])
        for a, b in zip(starts, ends):
            enter = float(ts[0]) if a == 0 else crossing(ts[a - 1], ts[a])
            leave = float(ts[last]) if b == last else crossing(ts[b + 1], ts[b])
            intervals.append((enter, leave))

    # A visit shorter than the sample spacing shows up as a local minimum of the
    # samples with g > 0; minimize g between the neighbouring samples to find it.
    for i in range(last + 1):
        if gv[i] <= 0 or (i > 0 and gv[i - 1] < gv[i]) or (i < last and gv[i + 1] < gv[i]):
            continue
        # g can only dip below zero between samples if it is small compared with how
        # much it changes from one sample to the next.
        change = max(abs(gv[i] - gv[max(i - 1, 0)]), abs(gv[min(i + 1, last)] - gv[i]))
        if gv[i] > 2.0 * change:
            continue
        lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, last)]
        res = minimize_scalar(g_of_t, bounds=(lo, hi), method="bounded", options={"xatol": 1e-13})
        if res.fun < min_g:
            min_g, min_g_time = float(res.fun), float(res.x)
        if res.fun <= 0:
            mid = float(res.x)
            enter = crossing(lo, mid) if g_of_t(lo) > 0 else float(lo)
            leave = crossing(hi, mid) if g_of_t(hi) > 0 else float(hi)
            intervals.append((enter, leave))

    intervals = merge_intervals(intervals)
    first = intervals[0][0] if intervals else np.nan

    viol = np.nan
    if domain is not None:
        viol = float(np.max(np.maximum(domain.inside_function(X), 0.0)))

    return TrajectoryCheck(np.asarray(x0), bool(intervals), first, min_g, min_g_time, viol, True,
                           sol.message, intervals)


def check_many(system, X0, target, time_horizon, domain=None, stop_at_first_hit=False, **kw):
    results = []
    for x0 in X0:
        r = check_trajectory(system, x0, target, time_horizon, domain=domain, **kw)
        results.append(r)
        if stop_at_first_hit and r.hit:
            break
    return results


def simulate(system, x0, time_horizon, n_points=1001, method="DOP853"):
    """Plain trajectory on a uniform grid (for plotting)."""
    t0, t1 = time_horizon
    ts = np.linspace(t0, t1, n_points)
    sol = solve_ivp(system.ode, (t0, t1), np.asarray(x0, float), method=method,
                    rtol=RTOL, atol=ATOL, t_eval=ts)
    return sol.t, sol.y.T, sol.success
