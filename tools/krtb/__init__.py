"""KRTB: reach-time bounds from principal Koopman eigenfunctions (the paper's method).

Run with `python -m tools.krtb.run --benchmark <ID>` from the repository root.
"""

from .bounds import ReachTimeBounds, compute_reach_time_bounds
from .eigenfunctions import Eigenfunctions, analytic, path_integral

__all__ = ["Eigenfunctions", "ReachTimeBounds", "analytic", "compute_reach_time_bounds", "path_integral"]
