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
"""Tests for optimization problem configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from unittest import mock

import pytest
from gemseo.doe import CustomDOE_Settings
from gemseo.optimization import NLOPT_COBYLA_Settings
from gemseo.optimization import OptimizationProblem
from gemseo.optimization import SLSQP_Settings
from gemseo.problem.optimization.rosenbrock import Rosenbrock
from gemseo.space import DesignSpace
from gemseo.util.testing.helper import assert_exception
from numpy import array
from numpy.testing import assert_allclose

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.benchmarker.optimization_worker import OptimizationWorker
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems import optimization_problem_configuration
from gemseo_benchmark.problems.optimization_problem_configuration import (
    OptimizationProblemConfiguration,
)
from tests.problems.conftest import check_default_starting_point
from tests.problems.conftest import check_description
from tests.problems.conftest import check_inconsistent_starting_points
from tests.problems.conftest import check_no_starting_point
from tests.problems.conftest import check_number_of_scalar_constraints
from tests.problems.conftest import check_set_starting_points_as_iterable
from tests.problems.conftest import (
    check_set_starting_points_as_iterable_with_wrong_shape,
)
from tests.problems.conftest import check_set_starting_points_as_non_2d_array
from tests.problems.conftest import check_set_starting_points_as_non_iterable
from tests.problems.conftest import check_set_starting_points_with_wrong_dimension
from tests.problems.conftest import check_starting_point_loading
from tests.problems.conftest import check_starting_points_default_generation
from tests.problems.conftest import check_starting_points_generation
from tests.problems.conftest import check_starting_points_saving
from tests.problems.conftest import check_undefined_target_values
from tests.problems.conftest import check_variable_space
from tests.problems.conftest import check_worker_type
from tests.problems.conftest import plot_data_profiles
from tests.problems.conftest import plot_data_profiles_with_max_eval_number
from tests.problems.conftest import test_compute_data_profiles_parametrize
from tests.problems.conftest import test_description_parametrize

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


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
    """A problem configuration."""
    return OptimizationProblemConfiguration("Problem", minimization_problem_creator)


def test_undefined_starting_points() -> None:
    """Check the absence of starting points."""
    variable_space = DesignSpace()
    variable_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)

    def create_problem() -> OptimizationProblem:
        """Create an optimization problem."""
        return OptimizationProblem(variable_space)

    check_no_starting_point(OptimizationProblemConfiguration("Problem", create_problem))


def test_default_starting_point(benchmarking_problem, design_space):
    """Check that the default starting point is properly set."""
    check_default_starting_point(benchmarking_problem, design_space)


def test_inconsistent_starting_points(minimization_problem_creator):
    """Check initialization with starting points of inadequate size."""
    check_inconsistent_starting_points(
        OptimizationProblemConfiguration, minimization_problem_creator
    )


def test_starting_points_iteration(minimization_problem_creator):
    """Check the iteration on starting points."""
    starting_points = array([[0.0, 0.0], [1.0, 1.0]])
    problem = OptimizationProblemConfiguration(
        "problem",
        minimization_problem_creator,
        doe_settings=CustomDOE_Settings(samples=starting_points),
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


def test_undefined_target_values(minimization_problem_creator):
    """Check undefined target values."""
    check_undefined_target_values(
        OptimizationProblemConfiguration("problem", minimization_problem_creator)
    )


def test_generate_default_starting_points(minimization_problem_creator):
    """Check the generation of the default number of starting points."""
    check_starting_points_default_generation(
        OptimizationProblemConfiguration, minimization_problem_creator
    )


def test_generate_starting_points(minimization_problem_creator):
    """Check the generation of starting points."""
    check_starting_points_generation(
        OptimizationProblemConfiguration, minimization_problem_creator
    )


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
            (
                "A problem depending on 2 bounded variables,"
                " with a nonlinear objective,"
                " subject to 1 linear equality constraint."
            ),
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
        OptimizationProblemConfiguration._get_description(
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
    target1.performance_measure = 1.0
    target2 = mock.Mock()
    target2.performance_measure = 0.0
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

    problem = OptimizationProblemConfiguration(
        "Problem",
        create,
        target_values_algorithms_configurations=algorithms_configurations,
        target_values_number=2,
    )
    assert isinstance(problem.target_values, TargetValues)


@pytest.fixture
def execute_algo() -> Iterator[mock.Mock]:
    """Spy on the function executing the algorithms to compute the target values.

    Yields:
        The spy of the function.
    """
    with mock.patch.object(
        optimization_problem_configuration,
        "execute_algo",
        wraps=optimization_problem_configuration.execute_algo,
    ) as spy:
        yield spy


class _ThreeIterationsSLSQPSettings(SLSQP_Settings):
    """SLSQP settings with three iterations by default."""

    _inherited_field_defaults: ClassVar[dict[str, Any]] = {"max_iter": 3}


def test_compute_target_values_settings_defaults(execute_algo) -> None:
    """Check that the target values use the settings model and its defaults."""
    problem = OptimizationProblemConfiguration("Rosenbrock", Rosenbrock)
    configurations = AlgorithmsConfigurations(
        AlgorithmConfiguration(_ThreeIterationsSLSQPSettings())
    )
    problem.compute_target_values(1, configurations)
    assert execute_algo.call_args_list
    for call in execute_algo.call_args_list:
        settings = call.kwargs["settings_model"]
        assert isinstance(settings, _ThreeIterationsSLSQPSettings)
        assert settings.max_iter == 3


def test_compute_target_values_instance_algorithm_options(execute_algo) -> None:
    """Check that the target values use the options specific to the instances."""
    problem = OptimizationProblemConfiguration("Rosenbrock", Rosenbrock)
    problem.starting_points = [array([0.0, 0.0]), array([0.5, 0.5])]
    configurations = AlgorithmsConfigurations(
        AlgorithmConfiguration(
            SLSQP_Settings(max_iter=10),
            instance_algorithm_options={
                "max_iter": lambda configuration, index: configuration.dimension + index
            },
        )
    )
    problem.compute_target_values(1, configurations)
    assert [
        call.kwargs["settings_model"].max_iter for call in execute_algo.call_args_list
    ] == [2, 3]


def test_compute_target_values_independent_settings(execute_algo) -> None:
    """Check that each execution uses its own settings.

    NLOPT_COBYLA sets its option `stop_crit_n_x` from the dimension of the problem,
    so that the user's settings would be changed by the first execution
    and the second one would use a wrong value.
    """
    settings = NLOPT_COBYLA_Settings(max_iter=10)
    configurations = AlgorithmsConfigurations(AlgorithmConfiguration(settings))
    for dimension in (2, 5):
        OptimizationProblemConfiguration(
            f"Rosenbrock {dimension}",
            lambda dimension=dimension: Rosenbrock(dimension),
        ).compute_target_values(1, configurations)

    used_settings = [
        call.kwargs["settings_model"] for call in execute_algo.call_args_list
    ]
    assert [settings.stop_crit_n_x for settings in used_settings] == [3, 6]
    assert used_settings[0] is not used_settings[1]
    assert all(used is not settings for used in used_settings)
    assert settings.model_dump(exclude_unset=True) == {"max_iter": 10}


def test_compute_target_values_without_instance_options(execute_algo) -> None:
    """Check that the settings are not validated again without instance options."""
    settings = SLSQP_Settings(max_iter=3)
    configurations = AlgorithmsConfigurations(AlgorithmConfiguration(settings))
    problem = OptimizationProblemConfiguration("Rosenbrock", Rosenbrock)
    with mock.patch.object(
        SLSQP_Settings, "model_validate", wraps=SLSQP_Settings.model_validate
    ) as model_validate:
        problem.compute_target_values(1, configurations)

    model_validate.assert_not_called()
    used_settings = execute_algo.call_args.kwargs["settings_model"]
    assert used_settings is not settings
    assert used_settings.max_iter == 3


def test_compute_target_values_invalid_instance_option(execute_algo, snapshot) -> None:
    """Check the error raised by invalid options specific to the instances.

    The errors of all the instances are reported before any optimization,
    even if the algorithm configuration with these options is the last one.
    """
    problem = OptimizationProblemConfiguration("Rosenbrock", Rosenbrock)
    problem.starting_points = [array([0.0, 0.0]), array([0.5, 0.5])]
    valid_configuration = AlgorithmConfiguration(SLSQP_Settings(), "A SLSQP")
    invalid_configuration = AlgorithmConfiguration(
        SLSQP_Settings(),
        "B SLSQP",
        instance_algorithm_options={"max_iter": lambda configuration, index: "five"},
    )
    configurations = AlgorithmsConfigurations(
        invalid_configuration, valid_configuration
    )
    assert list(configurations)[-1] is invalid_configuration
    with assert_exception(ValueError, snapshot):
        problem.compute_target_values(1, configurations)

    execute_algo.assert_not_called()


def test_starting_points_non_2d_array(benchmarking_problem):
    """Check the setting of starting points as a non 2-dimensional NumPy array."""
    check_set_starting_points_as_non_2d_array(benchmarking_problem)


def test_starting_points_non_iterable(benchmarking_problem):
    """Check the setting of starting points as a non-iterable object."""
    check_set_starting_points_as_non_iterable(benchmarking_problem)


def test_starting_points_wrong_dimension(benchmarking_problem):
    """Check the setting of starting points of the wrong dimension."""
    check_set_starting_points_with_wrong_dimension(benchmarking_problem)


def test_starting_points_iterable(benchmarking_problem):
    """Check the setting of starting points as an iterable of NumPy arrays."""
    check_set_starting_points_as_iterable(benchmarking_problem)


def test_starting_points_iterable_wrong_shape(benchmarking_problem):
    """Check the setting of starting points as an iterable of ill-shaped arrays."""
    check_set_starting_points_as_iterable_with_wrong_shape(benchmarking_problem)


def test_starting_points_generator(minimization_problem_creator) -> None:
    """Check the setting of starting points as a generator of NumPy arrays."""
    problem = OptimizationProblemConfiguration("Problem", minimization_problem_creator)
    starting_points = [array([0.0, 0.0]), array([1.0, 1.0])]
    problem.starting_points = (point for point in starting_points)
    assert_allclose(problem.starting_points, starting_points)


def test_targets_generator_accessor(benchmarking_problem):
    """Check the accessor to the targets generator."""
    assert benchmarking_problem.targets_generator is None


def test_target_values_maximization_initial(maximization_problem_creator) -> None:
    """Check the initial target values for a maximization problem."""
    problem = OptimizationProblemConfiguration(
        "Rosenbrock maximization",
        maximization_problem_creator,
        target_values=TargetValues([1, 2], [3, 4]),
    )
    assert problem.target_values.performance_measures == [1, 2]
    assert problem.target_values.infeasibility_measures == [3, 4]
    assert problem.minimization_target_values.performance_measures == [-1, -2]
    assert problem.minimization_target_values.infeasibility_measures == [3, 4]


def test_target_values_maximization_set(maximization_problem_creator) -> None:
    """Check the set target values for a maximization problem."""
    problem = OptimizationProblemConfiguration(
        "Rosenbrock maximization", maximization_problem_creator
    )
    problem.target_values = TargetValues([1, 2], [3, 4])
    assert problem.target_values.performance_measures == [1, 2]
    assert problem.target_values.infeasibility_measures == [3, 4]
    assert problem.minimization_target_values.performance_measures == [-1, -2]
    assert problem.minimization_target_values.infeasibility_measures == [3, 4]


@test_description_parametrize
def test_description(
    minimization_problem_creator, input_description, actual_description
):
    """Check the description."""
    check_description(
        OptimizationProblemConfiguration,
        minimization_problem_creator,
        input_description,
        actual_description,
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
    check_starting_points_saving(
        tmp_path, OptimizationProblemConfiguration, minimization_problem_creator
    )


def test_load_starting_points(tmp_path, minimization_problem_creator):
    """Check the loading of starting points."""
    check_starting_point_loading(
        tmp_path, OptimizationProblemConfiguration, minimization_problem_creator
    )


def test_dimension(benchmarking_problem):
    """Check the problem dimension."""
    assert benchmarking_problem.dimension == 2


@test_compute_data_profiles_parametrize
def test_compute_data_profile(
    minimization_problem_creator,
    target_values,
    algorithms_configurations,
    results,
    use_abscissa_log_scale,
    snapshot_matplotlib,
):
    """Check the computation of data profiles."""
    figure = plot_data_profiles(
        OptimizationProblemConfiguration,
        minimization_problem_creator,
        target_values,
        algorithms_configurations,
        results,
        use_abscissa_log_scale,
    )
    assert figure == snapshot_matplotlib


def test_compute_data_profile_max_eval_number(
    minimization_problem_creator,
    target_values,
    algorithms_configurations,
    results,
    snapshot_matplotlib,
):
    """Check the computation of data profiles when the evaluations number is limited."""
    figure = plot_data_profiles_with_max_eval_number(
        OptimizationProblemConfiguration,
        minimization_problem_creator,
        target_values,
        algorithms_configurations,
        results,
    )
    assert figure == snapshot_matplotlib(remove_text=True)


@pytest.mark.parametrize(
    ("problem_creator", "minimize_objective"),
    [("minimization_problem_creator", True), ("maximization_problem_creator", False)],
)
def test_minimize_objective(problem_creator, minimize_objective, request) -> None:
    """Check the minimization flag."""
    assert (
        OptimizationProblemConfiguration(
            "Problem", request.getfixturevalue(problem_creator)
        ).minimize_performance_measure
        is minimize_objective
    )


def test_variable_space(benchmarking_problem, design_space) -> None:
    """Check the variable space."""
    check_variable_space(benchmarking_problem, design_space)


def test_worker(benchmarking_problem) -> None:
    """Check the type of benchmarking worker."""
    check_worker_type(benchmarking_problem, OptimizationWorker)


def test_number_of_scalar_constraints(benchmarking_problem) -> None:
    """Check the number of scalar constraints."""
    check_number_of_scalar_constraints(benchmarking_problem, 2)
