"""Parsing of the string expressions used in benchmark configs.

Expressions are written in the state variables x1..xn with the usual
arithmetic, '^' or '**' for powers, and the functions listed in FUNCTIONS.
Anything else (unknown names, free parameters, x_{n+1}) is rejected at parse
time instead of silently evaluating to something wrong at run time.

A system may name constants and sub-expressions (e.g. a control law u = -K x);
expand() substitutes them textually, so every tool still receives equations in
x1..xn written in the config's own syntax.
"""

import re

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "tanh": sp.tanh,
    "atan": sp.atan,
    "atan2": sp.atan2,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "pi": sp.pi,
    "I": sp.I,  # imaginary unit, for complex eigenfunctions
}


def state_symbols(n):
    return sp.symbols(f"x1:{n + 1}", real=True)


def parse(expression, n, extra=()):
    """Parse one expression in x1..xn (and the extra symbol names, if any) into a sympy expression."""
    symbols = state_symbols(n) + tuple(sp.Symbol(name, real=True) for name in extra)
    local = {s.name: s for s in symbols}
    local.update(FUNCTIONS)
    text = expression.replace("^", "**")
    try:
        expr = parse_expr(
            text,
            local_dict=local,
            global_dict={"Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational, "Symbol": sp.Symbol},
            transformations=standard_transformations,
            evaluate=True,
        )
    except Exception as exc:  # sympy raises many exception types
        raise ValueError(f"cannot parse expression {expression!r}: {exc}") from exc
    unknown = expr.free_symbols - set(symbols)
    if unknown:
        names = ", ".join(sorted(s.name for s in unknown))
        allowed = ", ".join([f"x1..x{n}", *extra])
        raise ValueError(f"expression {expression!r} uses unknown symbols {names}; allowed: {allowed}")
    return expr


def _number(v):
    """A constant in plain positional notation (no exponent: dReach cannot read one)."""
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    return np.format_float_positional(float(v), trim="-")


def check_name(name, n):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError(f"invalid name {name!r}")
    if re.fullmatch(r"x\d+", name) or name in FUNCTIONS:
        raise ValueError(f"name {name!r} clashes with a state variable or a function")


def substitute(text, values):
    """Replace every whole-word occurrence of each name with '(value)'."""
    if not values:
        return text
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, values)) + r")\b")
    return pattern.sub(lambda m: f"({values[m.group(1)]})", text)


def expand(equations, n, parameters=None, definitions=None):
    """Equations in x1..xn with named constants and sub-expressions substituted.

    parameters maps names to numbers; definitions maps names to expressions in the
    states, the parameters and earlier definitions (in order). Each result is
    checked to parse in x1..xn only.
    """
    parameters, definitions = dict(parameters or {}), dict(definitions or {})
    for name in [*parameters, *definitions]:
        check_name(name, n)
    clash = set(parameters) & set(definitions)
    if clash:
        raise ValueError(f"names defined twice: {sorted(clash)}")
    values = {k: _number(v) for k, v in parameters.items()}
    known = list(parameters)
    for name, text in definitions.items():
        parse(text, n, extra=known)  # only states, parameters and earlier definitions
        values[name] = substitute(text, values)
        known.append(name)
    out = [substitute(e, values) for e in equations]
    for e in out:
        parse(e, n)
    return out


def parse_all(expressions, n):
    return [parse(e, n) for e in expressions]


def vectorize(exprs, n):
    """Return F(X) mapping an (m, n) array to an (m, k) array for k expressions."""
    symbols = state_symbols(n)
    f = sp.lambdify(symbols, list(exprs), modules="numpy", cse=True)
    k = len(exprs)

    def F(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((X.shape[0], k))
        for j, col in enumerate(f(*X.T)):
            out[:, j] = col  # broadcasts constant expressions
        return out

    return F


def scalar_function(expression, n):
    """Vectorized f(X) -> (m,) for a single expression."""
    F = vectorize([parse(expression, n)], n)
    return lambda X: F(X)[:, 0]
