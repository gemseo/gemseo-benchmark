# -*- coding: utf-8 -*-
# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for benchmarking reference problems."""
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import ones, zeros
from numpy.testing import assert_allclose
from pytest import raises

from problems.problem import Problem


def test_invalid_creator():
    """Check initialization with an invalid problem creator."""
    with raises(TypeError, match="Creator must return an OptimizationProblem"):
        Problem("A problem", lambda: None, [zeros(2)])


def test_missing_start_points():
    """Check initialization without starting points."""
    with raises(ValueError, match="The starting points, "
                                  "or their number and the name of the algorithm to "
                                  "generate them, "
                                  "must be passed"):
        Problem("Rosenbrock2D", Rosenbrock)


def test_wrong_start_points_type():
    """Check initialization with starting points of the wrong type."""
    with raises(TypeError, match="Starting points must be of type ndarray"):
        Problem("Rosenbrock2D", Rosenbrock, [[0.0, 0.0]])


def test_inconsistent_start_points():
    """Check initialization with starting points of inadequate size."""
    with raises(ValueError, match="Starting points must be 1-dimensional with size 2"):
        Problem("Rosenbrock2D", Rosenbrock, [zeros(3)])


def test_start_points_iteration():
    """Check the iteration on start points."""
    start_points = [zeros(2), ones(2)]
    problem = Problem("Rosenbrock2D", Rosenbrock, start_points)
    problem_instances = list(problem)
    assert len(problem_instances) == 2
    assert_allclose(problem_instances[0].design_space.get_current_x(), start_points[0])
    assert_allclose(problem_instances[1].design_space.get_current_x(), start_points[1])


def test_undefined_targets():
    """Check the access to undefined targets."""
    problem = Problem("Rosenbrock2D", Rosenbrock, [zeros(2)])
    with raises(ValueError, match="Benchmarking problem has no target"):
        problem.target_values


def test_generate_start_points():
    """Check the generation of starting points."""
    problem = Problem(
        "Rosenbrock2D", Rosenbrock, doe_algo_name="DiagonalDOE", doe_size=5
    )
    assert len(list(problem.start_points)) == 5
