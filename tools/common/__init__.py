"""Shared Python infrastructure: benchmark specs, sets, expressions, simulation, result records."""

from .results import VERDICTS, agrees_with_expected, make_result, write_result
from .sets import Ball, Box, LevelSet, Union, make_set, make_union
from .spec import Benchmark, System, list_benchmarks, load_benchmark

__all__ = [
    "VERDICTS",
    "Ball",
    "Benchmark",
    "Box",
    "LevelSet",
    "System",
    "Union",
    "agrees_with_expected",
    "list_benchmarks",
    "load_benchmark",
    "make_result",
    "make_set",
    "make_union",
    "write_result",
]
