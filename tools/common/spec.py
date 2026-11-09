"""Loading and validating benchmark configs (configs/benchmarks/<ID>.json)."""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import jsonschema
import numpy as np

from . import expr, jsonio, paths
from .sets import Box, make_union


@lru_cache(maxsize=None)
def _schema(name):
    return jsonio.read_json(paths.SCHEMA_DIR / name)


def validate_against_schema(data, schema_name):
    jsonschema.validate(data, _schema(schema_name))


@dataclass
class System:
    kind: str
    dim: int
    equations: list = field(default_factory=list)  # strings, nonlinear systems
    A: np.ndarray = None  # linear systems

    def __post_init__(self):
        if self.kind == "nonlinear":
            self.exprs = expr.parse_all(self.equations, self.dim)
        else:
            self.A = np.asarray(self.A, dtype=float)
            syms = expr.state_symbols(self.dim)
            self.exprs = [sum(self.A[i, j] * syms[j] for j in range(self.dim)) for i in range(self.dim)]
        self._F = expr.vectorize(self.exprs, self.dim)

    def rhs(self, X):
        """f evaluated row-wise on an (m, n) array."""
        return self._F(X)

    def ode(self, t, x):
        """f(t, x) for scipy.integrate.solve_ivp (autonomous)."""
        return self._F(x[None, :])[0]

    def time_reversed(self):
        """The system dx/dt = -f(x)."""
        if self.kind == "nonlinear":
            return System("nonlinear", self.dim, equations=[f"-({e})" for e in self.equations])
        return System("linear", self.dim, A=-self.A)


@dataclass
class Benchmark:
    """A loaded benchmark.

    `system` is always the forward vector field f as written in the config. For a
    backward benchmark the trajectories that define reachability follow -f; use
    `problem_system` to simulate them.
    """

    id: str
    description: str
    system: System
    direction: str
    domain: Box
    initial_set: object  # Set (Union if several)
    target_set: object
    time_horizon: tuple
    expected_result: str
    notes: list
    vis_dims: list
    path: Path
    raw: dict

    @property
    def dim(self):
        return self.system.dim

    def solver_config(self):
        """The config with named parameters and definitions expanded into the equations,
        for tools that read the JSON themselves (JuliaReach, CORA)."""
        if self.system.kind != "nonlinear":
            return self.raw
        return {**self.raw, "system": {"type": "nonlinear", "equations": list(self.system.equations)}}

    @property
    def problem_system(self):
        """The vector field along which the benchmark's trajectories evolve (f or -f)."""
        if self.direction == "forward":
            return self.system
        if not hasattr(self, "_reversed"):
            self._reversed = self.system.time_reversed()
        return self._reversed


def benchmark_path(benchmark_id):
    return paths.BENCHMARK_DIR / f"{benchmark_id}.json"


def list_benchmarks():
    return sorted(p.stem for p in paths.BENCHMARK_DIR.glob("*.json"))


def load_benchmark(id_or_path):
    """Load, validate and build a Benchmark from an ID or a path to its JSON file."""
    path = Path(id_or_path)
    if path.suffix != ".json":
        path = benchmark_path(str(id_or_path))
    data = jsonio.read_json(path)
    validate_against_schema(data, "benchmark.schema.json")

    if data["id"] != path.stem:
        raise ValueError(f"{path.name}: id {data['id']!r} does not match the file name")

    sys_cfg = data["system"]
    if sys_cfg["type"] == "nonlinear":
        n = len(sys_cfg["equations"])
        try:
            equations = expr.expand(sys_cfg["equations"], n, sys_cfg.get("parameters"), sys_cfg.get("definitions"))
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        system = System("nonlinear", n, equations=equations)
    else:
        A = np.asarray(sys_cfg["A"], dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"{path.name}: A must be square")
        n = A.shape[0]
        system = System("linear", n, A=A)

    domain = Box(data["domain"])
    if domain.dim != n:
        raise ValueError(f"{path.name}: domain has dimension {domain.dim}, system has {n}")

    t0, t1 = data["time_horizon"]
    if not t1 > t0:
        raise ValueError(f"{path.name}: empty time horizon {data['time_horizon']}")

    vis_dims = data.get("visualization", {}).get("dims", [[0, 1]] if n >= 2 else [])
    for pair in vis_dims:
        if max(pair) >= n:
            raise ValueError(f"{path.name}: visualization dims {pair} out of range for n={n}")

    return Benchmark(
        id=data["id"],
        description=data["description"],
        system=system,
        direction=data.get("direction", "forward"),
        domain=domain,
        initial_set=make_union(data["initial_sets"], n),
        target_set=make_union(data["target_sets"], n),
        time_horizon=(float(t0), float(t1)),
        expected_result=data["expected_result"],
        notes=list(data.get("notes", [])),
        vis_dims=vis_dims,
        path=path,
        raw=data,
    )
