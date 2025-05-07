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
"""Tests for optimization benchmarking problems."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Callable
from unittest import mock

import numpy
import pytest
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from matplotlib import pyplot
from matplotlib.testing.decorators import image_comparison
from numpy import ones
from numpy import zeros
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems.optimization_benchmarking_problem import (
    OptimizationBenchmarkingProblem,
)

if TYPE_CHECKING:
    from gemseo import OptimizationProblem


@pytest.fixture(scope="module")
def minimization_problem_creator(
    minimization_problem,
) -> Callable[[], OptimizationProblem]:
    """A minimization problem creator."""
    return lambda: minimization_problem


@pytest.fixture(scope="module")
def maximization_problem_creator(
    maximization_problem,
) -> Callable[[], OptimizationProblem]:
    """A maximization problem creator."""
    return lambda: maximization_problem


@pytest.fixture(scope="module")
def benchmarking_problem(minimization_problem_creator):
    """A benchmarking problem."""
    return OptimizationBenchmarkingProblem("Problem", minimization_problem_creator)


def test_default_starting_point(benchmarking_problem, minimization_problem):
    """Check that the default starting point is properly set."""
    starting_points = benchmarking_problem.starting_points
    assert len(starting_points) == 1
    assert (
        starting_points[0] == minimization_problem.design_space.get_current_value()
    ).all()


def test_inconsistent_starting_points(minimization_problem_creator):
    """Check initialization with starting points of inadequate size."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "A starting point must be a 1-dimensional NumPy array of size 2."
        ),
    ):
        OptimizationBenchmarkingProblem(
            "problem", minimization_problem_creator, [zeros(3)]
        )


def test_starting_points_iteration(minimization_problem_creator):
    """Check the iteration on start points."""
    starting_points = [zeros(2), ones(2)]
    problem = OptimizationBenchmarkingProblem(
        "problem", minimization_problem_creator, starting_points
    )
    problem_instances = list(problem)
    assert len(problem_instances) == 2
    assert_allclose(
        problem_instances[0].design_space.set_current_value.call_args_list[0][0][0],
        starting_points[0],
    )
    assert_allclose(
        problem_instances[1].design_space.set_current_value.call_args_list[1][0][0],
        starting_points[1],
    )


def test_undefined_targets(minimization_problem_creator):
    """Check the access to undefined targets."""
    problem = OptimizationBenchmarkingProblem(
        "problem", minimization_problem_creator, [zeros(2)]
    )
    with pytest.raises(
        ValueError, match=re.escape("The benchmarking problem has no target value.")
    ):
        _ = problem.target_values


def test_generate_starting_points(minimization_problem_creator):
    """Check the generation of starting points."""
    problem = OptimizationBenchmarkingProblem(
        "problem", minimization_problem_creator, doe_algo_name="DiagonalDOE", doe_size=5
    )
    assert len(problem.starting_points) == 5


def test_undefined_starting_points(minimization_problem_creator):
    """Check the access to nonexistent starting points."""
    opt_problem = minimization_problem_creator()
    opt_problem.design_space.has_current_value = False
    problem = OptimizationBenchmarkingProblem("problem", lambda: opt_problem)
    with pytest.raises(
        ValueError, match=re.escape("The benchmarking problem has no starting point.")
    ):
        _ = problem.starting_points


@pytest.mark.parametrize(
    (
        "dimension",
        "nonlinear_objective",
        "linear_equality_constraints",
        "linear_inequality_constraints",
        "nonlinear_equality_constraints",
        "nonlinear_inequality_constraints",
        "description",
    ),
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
        OptimizationBenchmarkingProblem._get_description(
            dimension,
            nonlinear_objective,
            linear_equality_constraints,
            linear_inequality_constraints,
            nonlinear_equality_constraints,
            nonlinear_inequality_constraints,
        )
        == description
    )


@pytest.fixture
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


@pytest.mark.parametrize("minimize_objective", [False, True])
def test_init_targets_computation(algorithms_configurations, minimize_objective):
    """Check the computation of targets at the problem creation."""

    def create() -> OptimizationProblem:
        problem = Rosenbrock()
        problem.minimize_objective = minimize_objective
        return problem

    problem = OptimizationBenchmarkingProblem(
        "Problem",
        create,
        target_values_algorithms_configurations=algorithms_configurations,
        target_values_number=2,
    )
    assert isinstance(problem.target_values, TargetValues)


message = (
    "The starting points shall be passed as (lines of) a 2-dimensional NumPy "
    "array, or as an iterable of 1-dimensional NumPy arrays."
)


def test_starting_points_non_2d_array(benchmarking_problem):
    """Check the setting of starting points as a non 2-dimensional NumPy array."""
    with pytest.raises(
        ValueError,
        match=re.escape(f"{message} A 1-dimensional NumPy array was passed."),
    ):
        benchmarking_problem.starting_points = zeros(2)


def test_starting_points_non_iterable(benchmarking_problem):
    """Check the setting of starting points as a non-iterable object."""
    starting_points = 0.0
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{message} The following type was passed: {type(starting_points)}."
        ),
    ):
        benchmarking_problem.starting_points = starting_points


def test_starting_points_wrong_dimension(benchmarking_problem):
    """Check the setting of starting points of the wrong dimension."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"{message} The number of columns (1) is different from the problem "
            "dimension (2)."
        ),
    ):
        benchmarking_problem.starting_points = zeros((3, 1))


@pytest.mark.parametrize("doe_options", [None, {}])
def test_get_starting_points(minimization_problem_creator, doe_options):
    """Check the computation of the starting points."""
    bench_problem = OptimizationBenchmarkingProblem(
        "Problem",
        minimization_problem_creator,
        doe_algo_name="DiagonalDOE",
        doe_options=doe_options,
    )
    assert len(bench_problem.starting_points) == 2


def test_targets_generator_accessor(benchmarking_problem):
    """Check the accessor to the targets generator."""
    assert benchmarking_problem.targets_generator is None


def test_target_values_maximization_initial(maximization_problem_creator) -> None:
    """Check the initial target values for a maximization problem."""
    problem = OptimizationBenchmarkingProblem(
        "Rosenbrock maximization",
        maximization_problem_creator,
        [],
        TargetValues([1, 2], [3, 4]),
    )
    assert problem.target_values.objective_values == [1, 2]
    assert problem.target_values.infeasibility_measures == [3, 4]
    assert problem.minimization_target_values.objective_values == [-1, -2]
    assert problem.minimization_target_values.infeasibility_measures == [3, 4]


def test_target_values_maximization_set(maximization_problem_creator) -> None:
    """Check the set target values for a maximization problem."""
    problem = OptimizationBenchmarkingProblem(
        "Rosenbrock maximization", maximization_problem_creator
    )
    problem.target_values = TargetValues([1, 2], [3, 4])
    assert problem.target_values.objective_values == [1, 2]
    assert problem.target_values.infeasibility_measures == [3, 4]
    assert problem.minimization_target_values.objective_values == [-1, -2]
    assert problem.minimization_target_values.infeasibility_measures == [3, 4]


@pytest.mark.parametrize(
    ("input_description", "description"),
    [
        ({}, "No description available."),
        (
            {"description": "A description of the problem."},
            "A description of the problem.",
        ),
    ],
)
def test_default_description(
    minimization_problem_creator, input_description, description
):
    """Check the default description of a problem."""
    assert (
        OptimizationBenchmarkingProblem(
            "problem", minimization_problem_creator, **input_description
        ).description
        == description
    )


def test_objective_name(benchmarking_problem, minimization_problem_creator):
    """Check the accessor to the objective name."""
    assert (
        benchmarking_problem.objective_name
        == minimization_problem_creator().objective.name
    )


def test_constraints_names(benchmarking_problem, minimization_problem_creator):
    """Check the accessor to the constraints names."""
    assert (
        benchmarking_problem.constraints_names
        == minimization_problem_creator().scalar_constraint_names
    )


def test_save_starting_points(tmp_path, minimization_problem_creator):
    """Check the saving of starting points."""
    starting_points = numpy.ones((3, 2))
    path = tmp_path / "starting_points.npy"
    OptimizationBenchmarkingProblem(
        "problem", minimization_problem_creator, starting_points=starting_points
    ).save_starting_points(path)
    assert_equal(numpy.load(path), starting_points)


def test_load_starting_points(tmp_path, minimization_problem_creator):
    """Check the loading of starting points."""
    starting_points = numpy.ones((3, 2))
    path = tmp_path / "starting_points.npy"
    numpy.save(path, starting_points)
    problem = OptimizationBenchmarkingProblem(
        "problem", minimization_problem_creator, starting_points=starting_points
    )
    problem.load_starting_point(path)
    assert_equal(problem.starting_points, starting_points)


def test_dimension(benchmarking_problem):
    """Check the accessor to the problem dimension."""
    assert benchmarking_problem.dimension == 2


@pytest.mark.parametrize(
    ("baseline_images", "use_iteration_log_scale"),
    [
        (
            [f"data_profiles[use_iteration_log_scale={use_iteration_log_scale}]"],
            use_iteration_log_scale,
        )
        for use_iteration_log_scale in [False, True]
    ],
)
@image_comparison(None, ["png"])
def test_compute_data_profile(
    baseline_images,
    minimization_problem_creator,
    target_values,
    algorithms_configurations,
    results,
    use_iteration_log_scale,
):
    """Check the computation of data profiles."""
    target_values.compute_target_hits_history = mock.Mock(
        return_value=[0, 0, 0, 1, 1, 2]
    )
    bench_problem = OptimizationBenchmarkingProblem(
        "Problem", minimization_problem_creator, target_values=target_values
    )
    pyplot.close("all")
    bench_problem.compute_data_profile(
        algorithms_configurations,
        results,
        use_iteration_log_scale=use_iteration_log_scale,
    )


@image_comparison(
    baseline_images=["data_profiles_max_eval_number"],
    remove_text=True,
    extensions=["png"],
)
def test_compute_data_profile_max_eval_number(
    minimization_problem_creator, target_values, algorithms_configurations, results
):
    """Check the computation of data profiles when the evaluations number is limited."""
    bench_problem = OptimizationBenchmarkingProblem(
        "Problem", minimization_problem_creator, target_values=target_values
    )
    target_values.compute_target_hits_history = mock.Mock(return_value=[0, 0, 0, 1])
    pyplot.close("all")
    bench_problem.compute_data_profile(
        algorithms_configurations, results, max_iteration_number=4
    )


@pytest.mark.parametrize(
    ("problem_creator", "minimize_objective"),
    [("minimization_problem_creator", True), ("maximization_problem_creator", False)],
)
def test_minimize_objective(problem_creator, minimize_objective, request) -> None:
    """Check the minimization flag."""
    assert (
        OptimizationBenchmarkingProblem(
            "Problem", request.getfixturevalue(problem_creator)
        ).minimize_performance_measure
        is minimize_objective
    )
