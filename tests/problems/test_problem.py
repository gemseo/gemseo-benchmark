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
from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import numpy
import pytest
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.results.performance_history import PerformanceHistory
from matplotlib import pyplot
from matplotlib.testing.decorators import image_comparison
from numpy import ones
from numpy import zeros
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from pytest import raises


def test_invalid_creator():
    """Check initialization with an invalid problem creator."""
    with raises(
        TypeError,
        match="optimization_problem_creator must return an OptimizationProblem.",
    ):
        Problem("A problem", lambda: None)


@pytest.fixture(scope="module")
def creator(problem):
    """An optimization problem creator."""
    return lambda: problem


@pytest.fixture(scope="module")
def benchmarking_problem(creator):
    """A benchmarking problem."""
    return Problem("Problem", creator)


def test_default_start_point(benchmarking_problem, problem):
    """Check that the default starting point is properly set."""
    start_points = benchmarking_problem.start_points
    assert len(start_points) == 1
    assert (start_points[0] == problem.design_space.get_current_value()).all()


def test_wrong_start_points_type(creator):
    """Check initialization with starting points of the wrong type."""
    with raises(
        TypeError,
        match="A starting point must be a 1-dimensional NumPy array of size 2.",
    ):
        Problem("problem", creator, [[0.0, 0.0]])


def test_inconsistent_start_points(creator):
    """Check initialization with starting points of inadequate size."""
    with raises(
        ValueError,
        match="A starting point must be a 1-dimensional NumPy array of size 2.",
    ):
        Problem("problem", creator, [zeros(3)])


def test_start_points_iteration(creator):
    """Check the iteration on start points."""
    start_points = [zeros(2), ones(2)]
    problem = Problem("problem", creator, start_points)
    problem_instances = list(problem)
    assert len(problem_instances) == 2
    assert_allclose(
        problem_instances[0].design_space.set_current_value.call_args_list[0][0][0],
        start_points[0],
    )
    assert_allclose(
        problem_instances[1].design_space.set_current_value.call_args_list[1][0][0],
        start_points[1],
    )


def test_undefined_targets(creator):
    """Check the access to undefined targets."""
    problem = Problem("problem", creator, [zeros(2)])
    with raises(ValueError, match="The benchmarking problem has no target value."):
        problem.target_values


def test_generate_start_points(creator):
    """Check the generation of starting points."""
    problem = Problem("problem", creator, doe_algo_name="DiagonalDOE", doe_size=5)
    assert len(problem.start_points) == 5


def test_undefined_start_points(creator):  # TODO: use creator
    """Check the access to nonexistent starting points."""
    opt_problem = mock.Mock(spec=OptimizationProblem)
    opt_problem.design_space = mock.Mock()
    opt_problem.design_space.has_current_value = mock.Mock(return_value=False)
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
        "description",
    ],
    [
        (
            1,
            False,
            0,
            0,
            0,
            0,
            "A problem depending on 1 bounded variable, with a linear objective.",
        ),
        (
            2,
            True,
            1,
            0,
            0,
            0,
            "A problem depending on 2 bounded variables,"
            " with a nonlinear objective,"
            " subject to 1 linear equality constraint.",
        ),
    ],
)
def test__get_description(
    dimension,
    nonlinear_objective,
    linear_equality_constraints,
    linear_inequality_constraints,
    nonlinear_equality_constraints,
    nonlinear_inequality_constraints,
    description,
):
    """Check the description getter."""
    assert (
        Problem._get_description(
            dimension,
            nonlinear_objective,
            linear_equality_constraints,
            linear_inequality_constraints,
            nonlinear_equality_constraints,
            nonlinear_inequality_constraints,
        )
        == description
    )


@pytest.fixture()
def results():
    """Paths to performance histories."""
    paths = [Path(__file__).parent / "history.json"]
    results = mock.Mock()
    results.get_paths = mock.Mock(return_value=paths)
    return results


@pytest.fixture(scope="function")
def target_values():
    """Target values."""
    # N.B. passing the configuration is required for the setter.
    target_values = mock.MagicMock(spec=TargetValues)
    target1 = mock.Mock()
    target1.objective_value = 1.0
    target2 = mock.Mock()
    target2.objective_value = 0.0
    target_values.__iter__.return_value = [target1, target2]
    target_values.__len__.return_value = 2
    return target_values


def test_init_targets_computation(creator, algorithms_configurations):
    """Check the computation of targets at the problem creation."""
    problem = Problem(
        "Problem",
        lambda: Rosenbrock(),
        target_values_algorithms_configurations=algorithms_configurations,
        target_values_number=2,
    )
    assert isinstance(problem.target_values, TargetValues)


@image_comparison(baseline_images=["histories"], remove_text=True, extensions=["png"])
def test_plot_histories(
    creator, target_values, algorithms_configurations, results, algorithm_configuration
):
    """Check the histories graphs."""
    problem = Problem("problem", creator, target_values=target_values)
    pyplot.close("all")
    problem.plot_histories(algorithms_configurations, results, show=False)
    results.get_paths.assert_called_once_with(algorithm_configuration.name, "problem")


@image_comparison(
    baseline_images=["histories_single_target"], remove_text=True, extensions=["png"]
)
def test_plot_histories_single_target(
    creator, algorithms_configurations, results, algorithm_configuration
):
    """Check the histories graphs for a problem with a single target value."""
    target_values = mock.MagicMock(spec=TargetValues)
    target = mock.Mock()
    target.objective_value = -1.0
    target_values.__iter__.return_value = [target]
    problem = Problem("problem", creator, target_values=target_values)
    pyplot.close("all")
    problem.plot_histories(algorithms_configurations, results, show=False)
    results.get_paths.assert_called_once_with(algorithm_configuration.name, "problem")


@image_comparison(
    baseline_images=["three_histories"], remove_text=True, extensions=["png"]
)
def test_plot_3_histories(
    tmpdir, creator, target_values, algorithms_configurations, algorithm_configuration
):
    """Check the histories graph for three histories."""
    histories = [
        PerformanceHistory([1.3, 1.0, 0.6, 0.4, 0.3]),
        PerformanceHistory([1.2, 1.0, 0.5, 0.4, 0.2]),
        PerformanceHistory([1.1, 0.7, 0.5, 0.1, 0.0]),
    ]
    paths = list()
    for index, history in enumerate(histories):
        path = tmpdir / f"history_{index + 1}.json"
        history.to_file(path)
        paths.append(path)
    results = mock.Mock()
    results.get_paths = mock.Mock(return_value=paths)
    problem = Problem("problem", creator, target_values=target_values)
    pyplot.close("all")
    problem.plot_histories(algorithms_configurations, results, show=False)
    results.get_paths.assert_called_once_with(algorithm_configuration.name, "problem")


@image_comparison(
    baseline_images=["infeasible_histories"], remove_text=True, extensions=["png"]
)
def test_plot_infeasible_histories(
    tmpdir, creator, target_values, algorithms_configurations, algorithm_configuration
):
    """Check the histories graph for histories with infeasible items."""
    histories = [
        PerformanceHistory([1.5, 1.1, 0.2], [1.0, 1.0, 0.0]),
        PerformanceHistory([1.0, 0.5, 0.1], [1.0, 0.0, 0.0]),
        PerformanceHistory([0.5, 0.2, 0.0], [1.0, 0.0, 0.0]),
    ]
    paths = list()
    for index, history in enumerate(histories):
        path = tmpdir / f"history_{index + 1}.json"
        history.to_file(path)
        paths.append(path)
    results = mock.Mock()
    results.get_paths = mock.Mock(return_value=paths)
    problem = Problem("problem", creator, target_values=target_values)
    pyplot.close("all")
    problem.plot_histories(algorithms_configurations, results, show=False)
    results.get_paths.assert_called_once_with(algorithm_configuration.name, "problem")


@image_comparison(
    baseline_images=["histories_plot_all"], remove_text=True, extensions=["png"]
)
def test_plot_histories_plot_all(
    creator, target_values, algorithms_configurations, results, algorithm_configuration
):
    """Check the plotting of all the histories."""
    problem = Problem("Problem", creator, target_values=target_values)
    pyplot.close("all")
    problem.plot_histories(
        algorithms_configurations, results, show=False, plot_all_histories=True
    )


@image_comparison(
    baseline_images=["histories_optimum"], remove_text=True, extensions=["png"]
)
def test_plot_histories_optimum(
    creator, target_values, algorithms_configurations, results, algorithm_configuration
):
    """Check the plotting the histories when the optimum is set."""
    problem = Problem("Problem", creator, target_values=target_values)
    problem.optimum = 0.0
    pyplot.close("all")
    problem.plot_histories(algorithms_configurations, results, show=False)


@image_comparison(
    baseline_images=["histories_max_eval_number"], remove_text=True, extensions=["png"]
)
def test_plot_histories_max_eval_number(
    creator, target_values, algorithms_configurations, results, algorithm_configuration
):
    """Check the plotting the histories when the number of evaluations is limited."""
    problem = Problem("Problem", creator, target_values=target_values)
    problem.optimum = 0.0
    pyplot.close("all")
    problem.plot_histories(
        algorithms_configurations, results, show=False, max_eval_number=4
    )


@image_comparison(
    baseline_images=["logarithmic_histories"], remove_text=True, extensions=["png"]
)
def test_use_log_scale(
    tmpdir, creator, target_values, algorithms_configurations, algorithm_configuration
):
    """Check the use of a logarithmic scale."""
    history = PerformanceHistory([1000, 100, 10, 1])
    path = tmpdir / "history.json"
    history.to_file(path)
    results = mock.Mock()
    results.get_paths = mock.Mock(return_value=[path])
    problem = Problem("problem", creator, target_values=TargetValues([100, 1]))
    pyplot.close("all")
    problem.plot_histories(
        algorithms_configurations, results, show=False, use_log_scale=True
    )
    results.get_paths.assert_called_once_with(algorithm_configuration.name, "problem")


message = (
    "The starting points shall be passed as (lines of) a 2-dimensional NumPy "
    "array, or as an iterable of 1-dimensional NumPy arrays."
)


def test_start_points_non_2d_array(benchmarking_problem):
    """Check the setting of starting points as a non 2-dimensional NumPy array."""
    with pytest.raises(
        ValueError,
        match=re.escape(f"{message} A 1-dimensional NumPy array was passed."),
    ):
        benchmarking_problem.start_points = zeros(2)


def test_start_points_non_iterable(benchmarking_problem):
    """Check the setting of starting points as a non-iterable object."""
    start_points = 0.0
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{message} The following type was passed: {type(start_points)}."
        ),
    ):
        benchmarking_problem.start_points = start_points


def test_start_points_wrong_dimension(benchmarking_problem):
    """Check the setting of starting points of the wrong dimension."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"{message} The number of columns (1) is different from the problem "
            "dimension (2)."
        ),
    ):
        benchmarking_problem.start_points = zeros((3, 1))


def test_get_start_points(creator):
    """Check the computation of the starting points."""
    bench_problem = Problem("Problem", creator, doe_algo_name="DiagonalDOE")
    assert len(bench_problem.start_points) == 2


def test_targets_generator_accessor(benchmarking_problem):
    """Check the accessor to the targets generator."""
    assert benchmarking_problem.targets_generator is None


def test_target_values_wrong_type(benchmarking_problem):
    """Check the setting of target values of the wrong type."""
    target_values = [1.0, 0.0]
    with pytest.raises(
        TypeError,
        match="Target values must be of type TargetValues. "
        f"Type {type(target_values)} was passed.",
    ):
        benchmarking_problem.target_values = target_values


@pytest.mark.parametrize(
    ["input_description", "description"],
    [
        (None, "No description available."),
        ("A description of the problem.", "A description of the problem."),
    ],
)
def test_default_description(creator, input_description, description):
    """Check the default description of a problem."""
    assert (
        Problem("problem", creator, description=input_description).description
        == description
    )


def test_objective_name(benchmarking_problem, creator):
    """Check the accessor to the objective name."""
    assert benchmarking_problem.objective_name == creator().objective.name


def test_constraints_names(benchmarking_problem, creator):
    """Check the accessor to the constraints names."""
    assert (
        benchmarking_problem.constraints_names
        == creator().get_scalar_constraint_names()
    )


def test_save_start_points(tmp_path, creator):
    """Check the saving of starting points."""
    start_points = numpy.ones((3, 2))
    path = tmp_path / "start_points.npy"
    Problem("problem", creator, start_points=start_points).save_start_points(path)
    assert_equal(numpy.load(path), start_points)


def test_load_start_points(tmp_path, creator):
    """Check the loading of starting points."""
    start_points = numpy.ones((3, 2))
    path = tmp_path / "start_points.npy"
    numpy.save(path, start_points)
    problem = Problem("problem", creator, start_points=start_points)
    problem.load_start_point(path)
    assert_equal(problem.start_points, start_points)


def test_dimension(benchmarking_problem):
    """Check the accessor to the problem dimension."""
    assert benchmarking_problem.dimension == 2


def test_compute_performance(problem, database):
    """Check the extraction of a history from a solved optimization problem."""
    problem.database = database
    obj_values, infeas_measures, feas_statuses = Problem.compute_performance(problem)
    assert obj_values == [2.0]
    assert infeas_measures == [1.0]
    assert feas_statuses == [False]


@image_comparison(
    baseline_images=["data_profiles"], remove_text=True, extensions=["png"]
)
def test_compute_data_profile(
    creator, target_values, algorithms_configurations, results
):
    """Check the computation of data profiles."""
    target_values.compute_target_hits_history = mock.Mock(
        return_value=[0, 0, 0, 1, 1, 2]
    )
    bench_problem = Problem("Problem", creator, target_values=target_values)
    pyplot.close("all")
    bench_problem.compute_data_profile(algorithms_configurations, results)


@image_comparison(
    baseline_images=["data_profiles_max_eval_number"],
    remove_text=True,
    extensions=["png"],
)
def test_compute_data_profile_max_eval_number(
    creator, target_values, algorithms_configurations, results
):
    """Check the computation of data profiles when the evaluations number is limited."""
    bench_problem = Problem("Problem", creator, target_values=target_values)
    target_values.compute_target_hits_history = mock.Mock(return_value=[0, 0, 0, 1])
    pyplot.close("all")
    bench_problem.compute_data_profile(
        algorithms_configurations, results, max_eval_number=4
    )
