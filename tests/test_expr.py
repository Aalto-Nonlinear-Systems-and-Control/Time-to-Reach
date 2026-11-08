import numpy as np
import pytest

from tools.common import expr


def test_power_and_functions():
    f = expr.scalar_function("x1^2 + sin(x2) - exp(0)", 2)
    assert f(np.array([[2.0, 0.0]]))[0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "text, n",
    [
        ("x1 + u", 2),  # free parameter
        ("x1 + x3", 2),  # state beyond the dimension
        ("foo(x1)", 1),  # unknown function
        ("x1 +* x2", 2),  # syntax error
    ],
)
def test_rejects_invalid_expressions(text, n):
    with pytest.raises(ValueError):
        expr.parse(text, n)


def test_x10_is_not_x1_followed_by_0():
    # util_global's string replacement turned x10 + x1 into 4.0 instead of 9
    f = expr.scalar_function("x10 + x1", 10)
    X = np.zeros((1, 10))
    X[0, 0], X[0, 9] = 1.0, 8.0
    assert f(X)[0] == pytest.approx(9.0)


def test_vectorize_broadcasts_constants():
    F = expr.vectorize(expr.parse_all(["x2", "1"], 2), 2)
    out = F(np.array([[0.0, 5.0], [1.0, 7.0]]))
    np.testing.assert_allclose(out, [[5.0, 1.0], [7.0, 1.0]])


def test_expand_substitutes_parameters_and_definitions():
    out = expr.expand(["x2", "u + a*x1^2"], 2, parameters={"a": 2, "k": 0.5}, definitions={"v": "k*x2", "u": "-v"})
    assert out == ["x2", "(-((0.5)*x2)) + (2)*x1^2"]
    # whole words only: 'a' inside 'tanh', 'k' inside a longer name, x10 vs x1
    assert expr.expand(["tanh(x1) + ka"], 1, parameters={"a": 1, "ka": 3}) == ["tanh(x1) + (3)"]
    assert expr.expand(["1e-05*x1"], 1) == ["1e-05*x1"]  # untouched text keeps its form
    assert expr.expand(["c*x1"], 1, parameters={"c": 1e-05}) == ["(0.00001)*x1"]  # no exponent (dReach)


@pytest.mark.parametrize(
    "parameters, definitions",
    [
        ({"x1": 1}, None),  # state variable
        ({"exp": 1}, None),  # function
        (None, {"u": "w", "w": "x1"}),  # refers to a later definition
        ({"u": 1}, {"u": "x1"}),  # defined twice
        ({"1a": 1}, None),  # not an identifier
    ],
)
def test_expand_rejects_bad_names(parameters, definitions):
    with pytest.raises(ValueError):
        expr.expand(["x1"], 1, parameters, definitions)
