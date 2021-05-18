from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import zeros
from pytest import raises

from problems.problem import Problem


def test_invalid_start_points():
    """Check initialization with invalid starting points"""
    with raises(ValueError):
        Problem("Rosenbrock2D", Rosenbrock, [zeros(2), zeros(3)])
