import numpy as np
import pytest

from tools.common.sets import Ball, Box, LevelSet, Union, make_set, make_union


def rng():
    return np.random.default_rng(0)


def test_box_membership_and_signed_distance():
    b = Box([[0, 1], [0, 2]])
    X = np.array([[0.5, 1.0], [2.0, 1.0], [0.0, 0.0]])
    np.testing.assert_array_equal(b.contains(X), [True, False, True])
    g = b.inside_function(X)
    assert g[0] == pytest.approx(-0.5) and g[1] == pytest.approx(1.0) and g[2] == pytest.approx(0.0)


def test_degenerate_box_is_a_point():
    b = Box([[0.8, 0.8], [0.3, 0.3]])
    assert b.contains(np.array([[0.8, 0.3]]))[0]
    assert not b.contains(np.array([[0.8, 0.31]]))[0]
    assert np.all(b.sample(5, rng()) == [0.8, 0.3])


def test_ball_samples_inside_and_on_boundary():
    s = Ball([1.0, -1.0], 0.1)
    assert np.all(s.contains(s.sample(500, rng())))
    np.testing.assert_allclose(np.linalg.norm(s.boundary_sample(50, rng()) - s.center, axis=1), 0.1)


def test_level_set_is_restricted_to_its_box():
    s = LevelSet("x1^2 + x2^2 - 1", [[0, 2], [-2, 2]])  # right half of the unit disc
    X = s.sample(300, rng())
    assert np.all(X[:, 0] >= 0) and np.all(np.hypot(X[:, 0], X[:, 1]) <= 1)
    assert not s.contains(np.array([[-0.5, 0.0]]))[0]  # inside the disc, outside the box
    assert s.boundary_touches_box()


def test_union_is_a_union_not_an_intersection():
    # Regression for the SOSBC encoding bug: disjoint targets were intersected (empty set).
    u = Union([Box([[0.6, 0.7], [0.2, 0.3]]), Box([[1.0, 1.1], [1.0, 1.1]])])
    np.testing.assert_array_equal(u.contains(np.array([[0.65, 0.25], [1.05, 1.05], [0.8, 0.5]])), [True, True, False])
    X = u.sample(100, rng())
    assert np.all(u.contains(X))
    assert np.any(X[:, 0] < 0.8) and np.any(X[:, 0] > 0.9)  # both components are sampled


def test_samples_of_different_sets_are_independent():
    # Regression: util_global reseeded the global RNG per set, giving identical offsets.
    g = rng()
    a = Box([[0, 1], [0, 1]]).sample(10, g)
    b = Box([[5, 6], [5, 6]]).sample(10, g)
    assert not np.allclose(a, b - 5)


def test_make_set_checks_dimension():
    with pytest.raises(ValueError):
        make_set({"type": "box", "bounds": [[0, 1]]}, 2)
    assert isinstance(make_union([{"type": "box", "bounds": [[0, 1]]}] * 2, 1), Union)


def test_invalid_box_bounds():
    with pytest.raises(ValueError):
        Box([[1, 0]])


def test_vertex_samples_of_a_box_are_vertices():
    b = Box([[0, 1], [2, 3], [-1, 1]])
    X = b.vertex_sample(200, rng())
    assert X.shape == (200, 3) and len(np.unique(X, axis=0)) == 8  # all vertices of the box
    assert all(set(np.unique(X[:, d])) == set(b.bounds[d]) for d in range(3))


def test_vertex_samples_of_a_level_set_lie_on_it_with_coordinates_at_the_bounds():
    # the pole-tip set of the cart-pole benchmarks: the extrema of log|psi| are at x1 = 1, x2 on the level set
    s = LevelSet("1.3 - x1 - sin(x2)", [[-1, 1], [-1, 1], [-2.5, 2.5]])
    X = s.vertex_sample(500, rng())
    assert X.shape == (500, 3) and np.all(s.contains(X))
    assert np.max(np.abs(s.value(X))) < 1e-9  # on f = 0, inside
    assert np.mean(X[:, 0] == 1) > 0.3 and np.mean(np.isin(X[:, 2], (-2.5, 2.5))) > 0.3


def test_vertex_samples_of_a_union_cover_its_members():
    u = Union([Box([[0.6, 0.7], [0.2, 0.3]]), Box([[1.0, 1.1], [1.0, 1.1]])])
    X = u.vertex_sample(20, rng())
    assert len(X) == 20 and np.all(u.contains(X)) and np.any(X[:, 0] < 0.8) and np.any(X[:, 0] > 0.9)

