"""Benchmark dynamics and sets in the form hj_reachability needs (JAX functions)."""

import hj_reachability as hj
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from tools.common import expr


def vector_field(system):
    """x -> f(x) as a JAX function: the vector field of the benchmark, unchanged."""
    xs = expr.state_symbols(system.dim)
    f = sp.lambdify(xs, list(system.exprs), modules="jax")

    def F(x):
        return jnp.stack([jnp.asarray(c, dtype=x.dtype) * jnp.ones((), x.dtype) for c in f(*x)])

    return F


class Autonomous(hj.ControlAndDisturbanceAffineDynamics):
    """dx/dt = F(x) without control or disturbance."""

    def __init__(self, F, n):
        self.F, self.n = F, n
        zero = hj.sets.Box(jnp.zeros(1), jnp.zeros(1))
        super().__init__("min", "max", zero, zero)

    def open_loop_dynamics(self, state, time):
        return self.F(state)

    def control_jacobian(self, state, time):
        return jnp.zeros((self.n, 1))

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((self.n, 1))


def _box(bounds):
    lo = jnp.asarray([b[0] for b in bounds], dtype=jnp.float32)
    hi = jnp.asarray([b[1] for b in bounds], dtype=jnp.float32)

    def g(x):  # exact signed distance: negative inside
        outside = jnp.linalg.norm(jnp.maximum(jnp.maximum(lo - x, x - hi), 0.0))
        depth = jnp.min(jnp.minimum(x - lo, hi - x))
        return jnp.where(outside > 0, outside, -depth)

    return g


def implicit_function(specs, n):
    """g(x) <= 0 exactly on the union of the sets (the membership convention of tools.common.sets):
    the signed distance of a box or a ball; for a level set {f <= 0} within its bounding box,
    max(f, signed distance of the box), so that the function is negative inside the set."""
    parts = []
    for spec in specs:
        if spec["type"] == "box":
            parts.append(_box(spec["bounds"]))
        elif spec["type"] == "ball":
            c, r = jnp.asarray(spec["center"], dtype=jnp.float32), spec["radius"]
            parts.append(lambda x, c=c, r=r: jnp.linalg.norm(x - c) - r)
        elif spec["type"] == "box_shell":  # outside the hole: max(box, inner_radius - |x - c|_inf)
            c = jnp.asarray(spec["center"], dtype=jnp.float32)
            box, r = _box([[ci - spec["outer_radius"], ci + spec["outer_radius"]] for ci in spec["center"]]), \
                spec["inner_radius"]
            parts.append(lambda x, c=c, box=box, r=r: jnp.maximum(box(x), r - jnp.max(jnp.abs(x - c))))
        elif spec["type"] == "level_set":
            f = sp.lambdify(expr.state_symbols(n), expr.parse(spec["function"], n), modules="jax")
            box = _box(spec["bounding_box"])
            parts.append(lambda x, f=f, box=box: jnp.maximum(f(*x), box(x)))
        else:
            raise ValueError(f"unknown set type {spec['type']!r}")
    return lambda x: jnp.min(jnp.stack([p(x) for p in parts]))


def on_grid(g, grid):
    flat = grid.states.reshape(-1, grid.states.shape[-1])
    return jax.vmap(g)(flat).reshape(grid.shape)


def target_values(specs, n, grid):
    """Initial values of the target on the grid: its implicit function at the nodes, as the level-set method takes it."""
    return on_grid(implicit_function(specs, n), grid)
