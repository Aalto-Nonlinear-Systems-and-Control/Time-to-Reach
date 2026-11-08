"""Branch correction for the argument of complex eigenfunction evaluations
(adapted from data_driven_reach_time, ddrv/common/angles.py)."""

import numpy as np


def correct_angle(a):
    """Unwrap np.angle output column-wise so that each column stays on one branch.

    np.angle returns values in (-pi, pi]. When the evaluations of an eigenfunction
    over a set straddle the negative real axis, a column holds values near +pi and
    near -pi although the phases are close, which corrupts the sup/inf used by the
    reach-time bounds. Whenever a column spans more than pi/2, its negative
    entries are shifted by 2*pi.
    """
    out = np.array(a, dtype=float, copy=True)
    span = np.abs(np.max(out, axis=0) - np.min(out, axis=0))
    for i in np.flatnonzero(span > 0.5 * np.pi):
        col = out[:, i]
        col[col < 0] += 2 * np.pi
    return out
