"""Reach-time bounds from Koopman eigenvalues and eigenfunction evaluations.

Adapted from data_driven_reach_time (ddrv/algo/reach_time_bounds.py, main at
3946687), with these changes:
- the magnitude bound intersects the bounds from both cones of eigenfunction
  weights (sum of weighted Re(lam) positive and negative); the original only
  used the negative cone when every eigenvalue had a negative real part,
- 'no information' (all bounds infinite) is reported as such instead of as
  unreachable,
- periodic windows are enumerated up to the time horizon, so the loop also
  terminates when the magnitude bound is unbounded above,
- intervals are clipped to the time horizon.

Along a trajectory psi_i(x(t)) = exp(lam_i t) psi_i(x(0)). Taking log-moduli,
Re(lam_i) t = log|psi_i(x(t))| - log|psi_i(x(0))|, which lies between
a_i = inf_target log|psi_i| - sup_initial log|psi_i| and
b_i = sup_target log|psi_i| - inf_initial log|psi_i|. Phases give the analogous
relation for Im(lam_i) t modulo 2 pi.
"""

from dataclasses import dataclass, field

import numpy as np

from . import lfp
from .angles import correct_angle

INF = float("inf")


def intersect(i1, i2):
    lo, hi = max(i1[0], i2[0]), min(i1[1], i2[1])
    return (lo, hi) if lo <= hi else None


@dataclass
class ReachTimeBounds:
    status: str  # "empty", "bounded", "periodic" or "no_information"
    intervals: list  # (lo, hi) pairs within the horizon (if one was given)
    unclipped: list  # the same before clipping to the horizon
    magnitude: tuple
    phase: tuple = None
    period: float = None
    details: dict = field(default_factory=dict)

    @property
    def empty(self):
        return self.status == "empty" or not self.intervals


def magnitude_bound(ef0, efF, lams):
    """Interval for t from the log-moduli; intersection over both weight cones."""
    mag0, magF = np.log(np.abs(ef0)), np.log(np.abs(efF))
    a = magF.min(axis=0) - mag0.max(axis=0)  # lower bound of Re(lam_i) t
    b = magF.max(axis=0) - mag0.min(axis=0)  # upper bound of Re(lam_i) t
    lr = np.real(lams)

    # Growing cone, lr.y > 0:   a.y / lr.y <= t <= b.y / lr.y
    grow_lo = lfp.solve(a, lr, maximize=True)
    grow_hi = lfp.solve(b, lr, maximize=False)
    # Decaying cone, lr.y < 0:  b.y / lr.y <= t <= a.y / lr.y, written with the
    # positive denominator (-lr).y
    decay_lo = lfp.solve(-b, -lr, maximize=True)
    decay_hi = lfp.solve(-a, -lr, maximize=False)

    grow = (grow_lo.value, grow_hi.value)
    decay = (decay_lo.value, decay_hi.value)
    both = intersect(grow, decay)
    details = {
        "growing_cone": grow,
        "decaying_cone": decay,
        "lfp_status": [r.status for r in (grow_lo, grow_hi, decay_lo, decay_hi)],
    }
    return both, details


def phase_bound(ef0, efF, lams):
    """Phase window for t (modulo the combined period) from the eigenfunction arguments."""
    ang0 = correct_angle(np.angle(ef0))
    angF = correct_angle(np.angle(efF))
    diff_min = angF.min(axis=0) - ang0.max(axis=0)
    diff_max = angF.max(axis=0) - ang0.min(axis=0)
    width = diff_max - diff_min
    li = np.imag(lams)

    def window(res, lo_diff, hi_diff):
        if res.weights is None:
            return (-INF, INF), None
        omega = float(np.dot(res.weights, li))
        if omega == 0:
            return (-INF, INF), None
        lo, hi = np.dot(res.weights, lo_diff) / omega, np.dot(res.weights, hi_diff) / omega
        return (min(lo, hi), max(lo, hi)), omega

    # Tightest window per unit frequency on each side of zero frequency.
    pos = lfp.solve(width, li, maximize=False)
    neg = lfp.solve(width, -li, maximize=False)
    win_pos, omega_pos = window(pos, diff_min, diff_max)
    _, omega_neg = window(neg, diff_max, diff_min)
    if omega_pos is not None and omega_neg is not None and not np.isclose(omega_pos, abs(omega_neg)):
        raise RuntimeError(f"phase frequencies differ: {omega_pos} vs {omega_neg}")
    details = {
        "lfp_status": [pos.status, neg.status],
        "weights": None if pos.weights is None else pos.weights.tolist(),
    }
    return win_pos, omega_pos, details


def periodic_windows(mag, phase, period, t_max, itol=1e-3):
    """Shift the phase window by multiples of the period and intersect with mag up to t_max."""
    lo = max(mag[0], 0.0)
    hi = min(mag[1], t_max)
    if lo > hi:
        return []
    k = np.ceil((lo - phase[1]) / period)  # first window whose upper end reaches lo
    out = []
    while True:
        w = (phase[0] + k * period, phase[1] + k * period)
        if w[0] > hi:
            break
        part = intersect(w, (lo, hi))
        if part is not None:
            if out and part[0] - out[-1][1] <= itol:
                out[-1] = (out[-1][0], max(out[-1][1], part[1]))
            else:
                out.append(part)
        k += 1
    return out


def compute_reach_time_bounds(ef0, efF, lams, horizon=None, itol=1e-3):
    """Reach-time bounds for trajectories from the initial set to the target set.

    Args:
        ef0, efF: eigenfunction values on samples of the initial and target sets,
            shape (num_points, num_eigenfunctions).
        lams: eigenvalues of the flow that the trajectories follow (negate them
            for a backward problem).
        horizon: optional (t0, T); intervals are clipped to it.
        itol: windows closer than this are merged.
    """
    lams = np.asarray(lams, dtype=complex)
    mag, mag_details = magnitude_bound(ef0, efF, lams)
    details = {"magnitude": mag_details}
    t_max = INF if horizon is None else float(horizon[1])

    def done(status, unclipped, phase=None, period=None):
        clipped = []
        for iv in unclipped:
            part = iv if horizon is None else intersect(iv, (float(horizon[0]), float(horizon[1])))
            if part is not None:
                clipped.append(part)
        if status != "no_information" and not clipped:
            status = "empty"
        return ReachTimeBounds(status, clipped, unclipped, mag, phase, period, details)

    if mag is None or mag[1] < 0:
        return done("empty", [])

    if np.all(np.isreal(lams)):
        if np.isinf(mag[0]) and np.isinf(mag[1]):
            return done("no_information", [(0.0, INF)])
        return done("bounded", [(max(mag[0], 0.0), mag[1])])

    phase, omega, phase_details = phase_bound(ef0, efF, lams)
    details["phase"] = phase_details
    if omega is None:
        if np.isinf(mag[0]) and np.isinf(mag[1]):
            return done("no_information", [(0.0, INF)])
        return done("bounded", [(max(mag[0], 0.0), mag[1])], phase)

    period = 2 * np.pi / abs(omega)
    if np.isinf(mag[1]) and np.isinf(t_max):
        # Unbounded and no horizon: report the first window; it repeats with the period.
        windows = periodic_windows(mag, phase, period, max(mag[0], 0.0) + period, itol)
        return done("periodic", windows[:1], phase, period)
    windows = periodic_windows(mag, phase, period, t_max, itol)
    return done("periodic" if np.isinf(mag[1]) else "bounded", windows, phase, period)
