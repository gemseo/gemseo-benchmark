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
from pathlib import Path

import pytest
from matplotlib import pyplot
from matplotlib.testing.decorators import image_comparison
from numpy import ones, zeros
from numpy.testing import assert_allclose
from pytest import raises

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.utils.py23_compat import mock
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.results.performance_history import PerformanceHistory


def test_invalid_creator():
    """Check initialization with an invalid problem creator."""
    with raises(
            TypeError,
            match="optimization_problem_creator must return an OptimizationProblem."
    ):
        Problem("A problem", lambda: None)


@pytest.fixture(scope="module")
def creator(problem):
    """An optimization problem creator."""
    return lambda: problem


def test_default_start_point(creator, problem):
    """Check that the default starting point is properly set."""
    start_points = Problem("problem", creator).start_points
    assert len(start_points) == 1
    assert (start_points[0] == problem.design_space.get_current_x()).all()


def test_wrong_start_points_type(creator):
    """Check initialization with starting points of the wrong type."""
    with raises(
            TypeError,
            match="A starting point must be a 1-dimensional NumPy array of size 2."
    ):
        Problem("problem", creator, [[0.0, 0.0]])


def test_inconsistent_start_points(creator):
    """Check initialization with starting points of inadequate size."""
    with raises(
            ValueError,
            match="A starting point must be a 1-dimensional NumPy array of size 2."
    ):
        Problem("problem", creator, [zeros(3)])


def test_start_points_iteration(creator):
    """Check the iteration on start points."""
    start_points = [zeros(2), ones(2)]
    problem = Problem("problem", creator, start_points)
    problem_instances = list(problem)
    assert len(problem_instances) == 2
    assert_allclose(
        problem_instances[0].design_space.set_current_x.call_args_list[0][0][0],
        start_points[0]
    )
    assert_allclose(
        problem_instances[1].design_space.set_current_x.call_args_list[1][0][0],
        start_points[1]
    )


def test_undefined_targets(creator):
    """Check the access to undefined targets."""
    problem = Problem("problem", creator, [zeros(2)])
    with raises(ValueError, match="The benchmarking problem has no target value."):
        problem.target_values


def test_generate_start_points(creator):
    """Check the generation of starting points."""
    problem = Problem(
        "problem", creator, doe_algo_name="DiagonalDOE", doe_size=5
    )
    assert len(problem.start_points) == 5


def test_undefined_start_points(creator):
    """Check the access to nonexistent starting points."""
    opt_problem = mock.Mock(spec=OptimizationProblem)
    opt_problem.design_space = mock.Mock()
    opt_problem.design_space.has_current_x = mock.Mock(return_value=False)
    problem = Problem("problem", lambda: opt_problem)
    with raises(ValueError, match="The benchmarking problem has no starting point."):
        problem.start_points


@pytest.mark.parametrize(
    [
        "dimension",
        "nonlinear_objective",
        "linear_equality_constraints",
        "linear_inequality_constraints",
        "nonlinear_equality_constraints",
        "nonlinear_inequality_constraints",
        "description"
    ],
    [
        (
                1, False, 0, 0, 0, 0,
                "A problem depending on 1 bounded variable, with a linear objective."
        ),
        (
                2, True, 1, 0, 0, 0,
                "A problem depending on 2 bounded variables,"
                " with a nonlinear objective,"
                " subject to 1 linear equality constraint."
        ),
    ]
)
def test__get_description(
        dimension, nonlinear_objective, linear_equality_constraints,
        linear_inequality_constraints, nonlinear_equality_constraints,
        nonlinear_inequality_constraints, description
):
    """Check the description getter."""
    assert Problem._get_description(
        dimension, nonlinear_objective, linear_equality_constraints,
        linear_inequality_constraints, nonlinear_equality_constraints,
        nonlinear_inequality_constraints
    ) == description


@pytest.fixture()
def results():
    """Paths to performance histories."""
    paths = [Path(__file__).parent / "history.json"]
    results = mock.Mock()
    results.get_paths = mock.Mock(return_value=paths)
    return results


@pytest.fixture(scope="module")
def target_values():
    """Target values."""
    # N.B. passing the specification is required for the setter.
    target_values = mock.MagicMock(spec=TargetValues)
    target1 = mock.Mock()
    target1.objective_value = 1.0
    target2 = mock.Mock()
    target2.objective_value = 0.0
    target_values.__iter__.return_value = [target1, target2]
    return target_values


@image_comparison(baseline_images=["histories"], remove_text=True, extensions=['png'])
def test_plot_histories(creator, target_values, results):
    """Check the histories graphs."""
    problem = Problem("problem", creator, target_values=target_values)
    algorithms_specs = {"algorithm": dict()}
    pyplot.close("all")
    problem.plot_histories(algorithms_specs, results, show=False)
    results.get_paths.assert_called_once_with("algorithm", "problem")


@image_comparison(
    baseline_images=["histories_single_target"], remove_text=True, extensions=['png']
)
def test_plot_histories_single_target(creator, results):
    """Check the histories graphs for a problem with a single target value."""
    target_values = mock.MagicMock(spec=TargetValues)
    target = mock.Mock()
    target.objective_value = -1.0
    target_values.__iter__.return_value = [target]
    problem = Problem("problem", creator, target_values=target_values)
    algorithms_specs = {"algorithm": dict()}
    pyplot.close("all")
    problem.plot_histories(algorithms_specs, results, show=False)
    results.get_paths.assert_called_once_with("algorithm", "problem")


@image_comparison(
    baseline_images=["three_histories"], remove_text=True, extensions=['png']
)
def test_plot_3_histories(tmpdir, creator, target_values):
    """Check the histories graph for three histories."""
    histories = [
        PerformanceHistory([1.3, 1.0, 0.6, 0.4, 0.3]),
        PerformanceHistory([1.2, 1.0, 0.5, 0.4, 0.2]),
        PerformanceHistory([1.1, 0.7, 0.5, 0.1, 0.0])
    ]
    paths = list()
    for index, history in enumerate(histories):
        path = tmpdir / f"history_{index + 1}.json"
        history.to_file(path)
        paths.append(path)
    results = mock.Mock()
    results.get_paths = mock.Mock(return_value=paths)
    problem = Problem("problem", creator, target_values=target_values)
    algorithms_specs = {"algorithm": dict()}
    pyplot.close("all")
    problem.plot_histories(algorithms_specs, results, show=False)
    results.get_paths.assert_called_once_with("algorithm", "problem")


@image_comparison(
    baseline_images=["infeasible_histories"], remove_text=True, extensions=["png"]
)
def test_plot_infeasible_histories(tmpdir, creator, target_values):
    """Check the histories graph for histories with infeasible items."""
    histories = [
        PerformanceHistory([1.5, 1.1, 0.2], [1.0, 1.0, 0.0]),
        PerformanceHistory([1.0, 0.5, 0.1], [1.0, 0.0, 0.0]),
        PerformanceHistory([0.5, 0.2, 0.0], [1.0, 0.0, 0.0])
    ]
    paths = list()
    for index, history in enumerate(histories):
        path = tmpdir / f"history_{index + 1}.json"
        history.to_file(path)
        paths.append(path)
    results = mock.Mock()
    results.get_paths = mock.Mock(return_value=paths)
    problem = Problem("problem", creator, target_values=target_values)
    algorithms_specs = {"algorithm": dict()}
    pyplot.close("all")
    problem.plot_histories(algorithms_specs, results, show=False)
    results.get_paths.assert_called_once_with("algorithm", "problem")
