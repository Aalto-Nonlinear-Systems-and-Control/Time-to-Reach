"""Set types used for initial and target sets (box, ball, box shell, level set, union).

Every set provides vectorized membership, a bounding box, sampling with an
explicit numpy Generator, and a continuous "inside function" g(X) with
g <= 0 exactly on the set. g is used for event detection during simulation and
for reporting margins; for boxes and balls it is the exact Euclidean signed
distance (outside), for level sets it is the function value.
"""

from dataclasses import dataclass, field

import numpy as np

from . import expr


def _as_bounds(bounds):
    b = np.asarray(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"bounds must be a list of [lower, upper] pairs, got {bounds!r}")
    if np.any(b[:, 0] > b[:, 1]):
        raise ValueError(f"lower bound exceeds upper bound in {bounds!r}")
    return b


class Set:
    dim: int
    kind = "set"

    def contains(self, X):
        return self.inside_function(X) <= 0

    def inside_function(self, X):
        raise NotImplementedError

    def bounding_box(self):
        raise NotImplementedError

    def sample(self, n, rng):
        raise NotImplementedError

    def boundary_sample(self, n, rng):
        """Points on (or very near) the boundary; used to catch extreme trajectories."""
        raise NotImplementedError

    def vertex_sample(self, n, rng):
        """Points towards the vertices of the bounding box, which uniform samples miss in high dimension."""
        raise NotImplementedError

    def margin_kind(self):
        return "euclidean_distance"


@dataclass
class Box(Set):
    bounds: np.ndarray
    kind = "box"

    def __post_init__(self):
        self.bounds = _as_bounds(self.bounds)
        self.dim = self.bounds.shape[0]

    @property
    def lower(self):
        return self.bounds[:, 0]

    @property
    def upper(self):
        return self.bounds[:, 1]

    def inside_function(self, X):
        X = np.atleast_2d(X)
        below = self.lower - X
        above = X - self.upper
        excess = np.maximum(np.maximum(below, above), 0.0)
        outside = np.linalg.norm(excess, axis=1)
        # inside: negative distance to the nearest face
        depth = np.min(np.minimum(X - self.lower, self.upper - X), axis=1)
        return np.where(outside > 0, outside, -depth)

    def bounding_box(self):
        return self.bounds.copy()

    def sample(self, n, rng):
        return self.lower + (self.upper - self.lower) * rng.random((n, self.dim))

    def corners(self, max_corners=4096):
        free = np.flatnonzero(self.upper > self.lower)
        if 2 ** len(free) > max_corners:
            return np.empty((0, self.dim))
        pts = np.tile(self.lower, (2 ** len(free), 1))
        for k in range(2 ** len(free)):
            for j, d in enumerate(free):
                if (k >> j) & 1:
                    pts[k, d] = self.upper[d]
        return pts

    def boundary_sample(self, n, rng):
        pts = self.sample(n, rng)
        free = np.flatnonzero(self.upper > self.lower)
        if len(free) == 0:
            return pts[:1]
        face = rng.choice(free, size=n)
        side = rng.random(n) < 0.5
        pts[np.arange(n), face] = np.where(side, self.lower[face], self.upper[face])
        return np.vstack([self.corners(), pts])

    def vertex_sample(self, n, rng):
        return np.where(rng.random((n, self.lower.size)) < 0.5, self.lower, self.upper)


@dataclass
class Ball(Set):
    center: np.ndarray
    radius: float
    kind = "ball"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.radius = float(self.radius)
        if self.radius <= 0:
            raise ValueError("ball radius must be positive")
        self.dim = self.center.shape[0]

    def inside_function(self, X):
        X = np.atleast_2d(X)
        return np.linalg.norm(X - self.center, axis=1) - self.radius

    def bounding_box(self):
        return np.stack([self.center - self.radius, self.center + self.radius], axis=1)

    def _directions(self, n, rng):
        v = rng.standard_normal((n, self.dim))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def sample(self, n, rng):
        r = self.radius * rng.random(n) ** (1.0 / self.dim)
        return self.center + self._directions(n, rng) * r[:, None]

    def boundary_sample(self, n, rng):
        return self.center + self.radius * self._directions(n, rng)


@dataclass
class BoxShell(Set):
    """{x : inner_radius <= ||x - center||_inf <= outer_radius}: the box of half-width outer_radius around the
    center without the open box of half-width inner_radius.

    A target around an equilibrium that excludes the equilibrium itself, as the time-to-reach bounds assume
    (x_e not in X_F). Interior and vertex samples are those of the outer box, drawn with the same random numbers;
    the rare samples in the hole are drawn again. boundary_sample covers the outer faces only: a sampler that
    needs the inner faces (where the principal eigenfunctions, which vanish at the center, are smallest) adds
    their points itself (tools.krtb.run.equilibrium_sample).
    """
    center: np.ndarray
    inner_radius: float
    outer_radius: float
    kind = "box_shell"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.inner_radius, self.outer_radius = float(self.inner_radius), float(self.outer_radius)
        if not 0 < self.inner_radius < self.outer_radius:
            raise ValueError("a box shell needs 0 < inner_radius < outer_radius")
        self.dim = self.center.shape[0]
        self.outer = Box(np.stack([self.center - self.outer_radius, self.center + self.outer_radius], axis=1))

    def inside_function(self, X):
        # outside the outer box: its Euclidean distance; inside: minus the distance to the nearest face, outer
        # or inner, so that g <= 0 exactly on the shell and g > 0 in the hole
        X = np.atleast_2d(X)
        hole = self.inner_radius - np.abs(X - self.center).max(axis=1)
        return np.maximum(self.outer.inside_function(X), hole)

    def bounding_box(self):
        return self.outer.bounding_box()

    def sample(self, n, rng):
        pts = self.outer.sample(n, rng)
        hole = ~self.contains(pts)
        while hole.any():
            pts[hole] = self.outer.sample(int(hole.sum()), rng)
            hole = ~self.contains(pts)
        return pts

    def boundary_sample(self, n, rng):
        return self.outer.boundary_sample(n, rng)

    def vertex_sample(self, n, rng):
        return self.outer.vertex_sample(n, rng)


@dataclass
class LevelSet(Set):
    function: str
    box: np.ndarray
    kind = "level_set"
    _f: object = field(default=None, repr=False)

    def __post_init__(self):
        self.box = _as_bounds(self.box)
        self.dim = self.box.shape[0]
        self._f = expr.scalar_function(self.function, self.dim)
        self._box_set = Box(self.box)

    def value(self, X):
        return self._f(np.atleast_2d(X))

    def inside_function(self, X):
        X = np.atleast_2d(X)
        # Outside the bounding box the set is empty by definition.
        return np.maximum(self.value(X), np.maximum(self._box_set.inside_function(X), 0.0))

    def bounding_box(self):
        return self.box.copy()

    def margin_kind(self):
        return "function_value"

    def sample(self, n, rng, batch=20000, max_batches=500):
        found = []
        count = 0
        for _ in range(max_batches):
            cand = self._box_set.sample(batch, rng)
            keep = cand[self.value(cand) <= 0]
            found.append(keep)
            count += len(keep)
            if count >= n:
                break
        pts = np.vstack(found)
        if len(pts) < n:
            raise ValueError(
                f"level set {self.function!r} is empty or tiny inside its bounding box "
                f"(found {len(pts)} of {n} samples)"
            )
        return pts[:n]

    def boundary_sample(self, n, rng):
        """Samples near f = 0, obtained by bisection between inside and outside points."""
        inside = self.sample(n, rng)
        outside = self._box_set.sample(4 * n, rng)
        outside = outside[self.value(outside) > 0][:n]
        m = min(len(inside), len(outside))
        a, b = inside[:m], outside[:m]
        for _ in range(40):
            mid = 0.5 * (a + b)
            in_mid = self.value(mid) <= 0
            a = np.where(in_mid[:, None], mid, a)
            b = np.where(in_mid[:, None], b, mid)
        return a

    def vertex_sample(self, n, rng, at_bound=0.75, batch=20000, max_batches=500):
        """Most coordinates at a bound of the box, the others free, then one coordinate bisected onto f = 0."""
        lo, hi = self.box[:, 0], self.box[:, 1]
        found = []
        count = 0
        for _ in range(max_batches):
            x = self._box_set.sample(batch, rng)
            bound = np.where(rng.random(x.shape) < 0.5, lo, hi)
            x = np.where(rng.random(x.shape) < at_bound, bound, x)
            j = rng.integers(self.dim, size=batch)
            a, b = x.copy(), x.copy()  # the two ends of coordinate j
            a[np.arange(batch), j], b[np.arange(batch), j] = lo[j], hi[j]
            in_a, in_b = self.value(a) <= 0, self.value(b) <= 0
            keep = in_a != in_b
            inside = np.where(in_a[:, None], a, b)[keep]
            outside = np.where(in_a[:, None], b, a)[keep]
            for _ in range(40):
                mid = 0.5 * (inside + outside)
                in_mid = self.value(mid) <= 0
                inside = np.where(in_mid[:, None], mid, inside)
                outside = np.where(in_mid[:, None], outside, mid)
            found.append(inside)
            count += len(inside)
            if count >= n:
                return np.vstack(found)[:n]
        raise ValueError(f"level set {self.function!r}: found {count} of {n} samples towards the vertices of its box")

    def boundary_touches_box(self, n=4000, rng=None):
        """True if the sublevel set reaches the bounding-box boundary (box may cut the set)."""
        rng = rng or np.random.default_rng(0)
        pts = self._box_set.boundary_sample(n, rng)
        return bool(np.any(self.value(pts) <= 0))


@dataclass
class Union(Set):
    sets: list
    kind = "union"

    def __post_init__(self):
        if not self.sets:
            raise ValueError("union of zero sets")
        dims = {s.dim for s in self.sets}
        if len(dims) != 1:
            raise ValueError(f"sets in a union have different dimensions {dims}")
        self.dim = dims.pop()

    def inside_function(self, X):
        return np.min(np.stack([s.inside_function(X) for s in self.sets]), axis=0)

    def bounding_box(self):
        boxes = np.stack([s.bounding_box() for s in self.sets])
        return np.stack([boxes[:, :, 0].min(0), boxes[:, :, 1].max(0)], axis=1)

    def _split(self, n):
        k = len(self.sets)
        return [n // k + (1 if i < n % k else 0) for i in range(k)]

    def sample(self, n, rng):
        return np.vstack([s.sample(m, rng) for s, m in zip(self.sets, self._split(n)) if m > 0])

    def boundary_sample(self, n, rng):
        return np.vstack([s.boundary_sample(m, rng) for s, m in zip(self.sets, self._split(n)) if m > 0])

    def vertex_sample(self, n, rng):
        return np.vstack([s.vertex_sample(m, rng) for s, m in zip(self.sets, self._split(n)) if m > 0])

    def margin_kind(self):
        kinds = {s.margin_kind() for s in self.sets}
        return kinds.pop() if len(kinds) == 1 else "mixed"


def make_set(spec, n):
    """Build a Set from its config dictionary; n is the state dimension."""
    kind = spec["type"]
    if kind == "box":
        s = Box(spec["bounds"])
    elif kind == "ball":
        s = Ball(spec["center"], spec["radius"])
    elif kind == "box_shell":
        s = BoxShell(spec["center"], spec["inner_radius"], spec["outer_radius"])
    elif kind == "level_set":
        s = LevelSet(spec["function"], spec["bounding_box"])
    else:
        raise ValueError(f"unknown set type {kind!r}")
    if s.dim != n:
        raise ValueError(f"{kind} set has dimension {s.dim}, system has {n}")
    return s


def make_union(specs, n):
    sets = [make_set(s, n) for s in specs]
    return sets[0] if len(sets) == 1 else Union(sets)
