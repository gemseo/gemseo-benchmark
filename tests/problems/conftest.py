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

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any
from unittest import mock

import matplotlib.pyplot
import numpy
import pytest
from numpy import zeros
from numpy.testing import assert_equal

from gemseo_benchmark.data_profiles.target_values import TargetValues

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from gemseo.algos.design_space import DesignSpace

    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.benchmarker.base_worker import BaseWorker
    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )
    from gemseo_benchmark.results.results import Results


def check_no_starting_point(
    problem_configuration: BaseProblemConfiguration,
) -> None:
    """Check that there is no starting point.

    Args:
        problem_configuration: The problem configuration.
    """
    with pytest.raises(
        ValueError, match=re.escape("The problem configuration has no starting point.")
    ):
        problem_configuration.starting_points  # noqa: B018


def check_default_starting_point(
    problem_configuration: BaseProblemConfiguration, variable_space: DesignSpace
) -> None:
    """Check the default starting point.

    Args:
        problem_configuration: The problem configuration.
        variable_space: The variable space of the problem configuration.
    """
    (starting_point,) = problem_configuration.starting_points
    assert (starting_point == variable_space.get_current_value()).all()


def check_inconsistent_starting_points(
    problem_configuration_class: type[BaseProblemConfiguration],
    create_problem: Callable,
    *args: Any,
) -> None:
    """Check initialization with starting points of inadequate size.

    Args:
        problem_configuration_class: The class of problem configuration.
        create_problem: A function that creates problems.
        *args: Positional arguments for the class initialization.
    """
    with pytest.raises(
        ValueError,
        match=re.escape(
            "A starting point must be a 1-dimensional NumPy array of size 2."
        ),
    ):
        problem_configuration_class(
            "Problem", create_problem, *args, starting_points=[numpy.zeros(3)]
        )


def check_undefined_target_values(
    problem_configuration: BaseProblemConfiguration,
) -> None:
    """Check that there is no target value.

    Args:
        problem_configuration: The problem configuration.
    """
    with pytest.raises(
        ValueError, match=re.escape("The problem configuration has no target value.")
    ):
        problem_configuration.target_values  # noqa: B018


def __check_starting_points_generation(
    problem_configuration_class: type[BaseProblemConfiguration],
    create_problem: Callable,
    input_doe_size: int | None,
    actual_doe_size: int,
    *args: Any,
) -> None:
    """Check the generation of the starting points.

    Args:
        problem_configuration_class: The class of problem configuration.
        create_problem: A function that creates problems.
        input_doe_size: The input number of starting points.
        actual_doe_size: The actual number of starting points.
        *args: Positional arguments for the class initialization.
    """
    assert (
        len(
            problem_configuration_class(
                "Problem",
                create_problem,
                *args,
                doe_algo_name="DiagonalDOE",
                doe_size=input_doe_size,
            ).starting_points
        )
        == actual_doe_size
    )


def check_starting_points_default_generation(
    problem_configuration_class: type[BaseProblemConfiguration],
    create_problem: Callable,
    *args: Any,
) -> None:
    """Check the generation of the starting points.

    Args:
        problem_configuration_class: The class of problem configuration.
        create_problem: A function that creates problems.
        *args: Positional arguments for the class initialization.
    """
    __check_starting_points_generation(
        problem_configuration_class, create_problem, None, 2, *args
    )


def check_starting_points_generation(
    problem_configuration_class: type[BaseProblemConfiguration],
    create_problem: Callable,
    *args: Any,
) -> None:
    """Check the generation of the starting points.

    Args:
        problem_configuration_class: The class of problem configuration.
        create_problem: A function that creates problems.
        *args: Positional arguments for the class initialization.
    """
    __check_starting_points_generation(
        problem_configuration_class, create_problem, 3, 3, *args
    )


message = (
    "The starting points shall be passed as (lines of) a 2-dimensional NumPy "
    "array, or as an iterable of 1-dimensional NumPy arrays."
)


def check_set_starting_points_as_non_2d_array(
    problem_configuration: BaseProblemConfiguration,
) -> None:
    """Check the setting of starting points as a non 2-dimensional NumPy array.

    Args:
        problem_configuration: The problem configuration.
    """
    with pytest.raises(
        ValueError,
        match=re.escape(f"{message} A 1-dimensional NumPy array was passed."),
    ):
        problem_configuration.starting_points = zeros(2)


def check_set_starting_points_as_non_iterable(
    problem_configuration: BaseProblemConfiguration,
) -> None:
    """Check the setting of starting points as a non-iterable.

    Args:
        problem_configuration: The problem configuration.
    """
    with pytest.raises(
        TypeError,
        match=re.escape(f"{message} The following type was passed: {float}."),
    ):
        problem_configuration.starting_points = 0.0


def check_set_starting_points_with_wrong_dimension(
    problem_configuration: BaseProblemConfiguration,
) -> None:
    """Check the setting of starting points of the wrong dimension.

    Args:
        problem_configuration: The problem configuration.
    """
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"{message} The number of columns (1) is different from the problem "
            "dimension (2)."
        ),
    ):
        problem_configuration.starting_points = numpy.zeros((3, 1))


test_description_parametrize = pytest.mark.parametrize(
    ("input_description", "actual_description"),
    [
        ({}, "No description available."),
        (
            {"description": "A description of the problem."},
            "A description of the problem.",
        ),
    ],
)


@test_description_parametrize
def check_description(
    problem_configuration_class: type[BaseProblemConfiguration],
    create_problem: Callable,
    input_description: Mapping[str, str],
    actual_description: str,
    *args: Any,
) -> None:
    """Check the description.

    Args:
        problem_configuration_class: The class of problem configuration.
        create_problem: A function that creates problems.
        input_description: The input description.
        actual_description: The actual description.
        *args: Positional arguments for the class initialization.
    """
    assert (
        problem_configuration_class(
            "Problem", create_problem, *args, **input_description
        ).description
        == actual_description
    )


def check_starting_points_saving(
    tmp_path,
    problem_configuration_class: type[BaseProblemConfiguration],
    create_problem: Callable,
    *args: Any,
) -> None:
    """Check the saving of the starting points.

    Args:
        tmp_path: The path to a temporary directory.
        problem_configuration_class: The class of problem configuration.
        create_problem: A function that creates problems.
        *args: Positional arguments for the class initialization.
    """
    starting_points = numpy.ones((3, 2))
    path = tmp_path / "starting_points.npy"
    problem_configuration_class(
        "Problem", create_problem, *args, starting_points=starting_points
    ).save_starting_points(path)
    assert_equal(numpy.load(path), starting_points)


def check_starting_point_loading(
    tmp_path,
    problem_configuration_class: type[BaseProblemConfiguration],
    create_problem: Callable,
    *args: Any,
) -> None:
    """Check the loading of starting points.

    Args:
        tmp_path: The path to a temporary directory.
        problem_configuration_class: The class of problem configuration.
        create_problem: A function that creates problems.
        *args: Positional arguments for the class initialization.
    """
    starting_points = numpy.ones((3, 2))
    path = tmp_path / "starting_points.npy"
    numpy.save(path, starting_points)
    problem = problem_configuration_class(
        "problem", create_problem, *args, starting_points=starting_points
    )
    problem.load_starting_point(path)
    assert_equal(problem.starting_points, starting_points)


test_compute_data_profiles_parametrize = pytest.mark.parametrize(
    ("baseline_images", "use_abscissa_log_scale"),
    [
        (
            [f"data_profiles[use_abscissa_log_scale={use_abscissa_log_scale}]"],
            use_abscissa_log_scale,
        )
        for use_abscissa_log_scale in [False, True]
    ],
)


def check_data_profiles_computation(
    problem_configuration_class: type[BaseProblemConfiguration],
    create_problem: Callable,
    target_values: TargetValues,
    algorithm_configurations: AlgorithmsConfigurations,
    results: Results,
    use_abscissa_log_scale: bool,
    *args: Any,
) -> None:
    """Check the computation of data profiles.

    Args:
        problem_configuration_class: The class of problem configuration.
        create_problem: A function that creates problems.
        target_values: The target values.
        algorithm_configurations: The algorithm configurations.
        results: The benchmarking results.
        use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
        *args: Positional arguments for the class initialization.
    """
    matplotlib.pyplot.close("all")
    target_values.compute_target_hits_history = mock.Mock(
        return_value=[0, 0, 0, 1, 1, 2]
    )
    problem_configuration_class(
        "Problem", create_problem, *args, target_values=target_values
    ).compute_data_profile(
        algorithm_configurations,
        results,
        use_abscissa_log_scale=use_abscissa_log_scale,
    )


def check_data_profiles_computation_with_max_eval_number(
    problem_configuration_class: type[BaseProblemConfiguration],
    create_problem: Callable,
    target_values: TargetValues,
    algorithm_configurations: AlgorithmsConfigurations,
    results: Results,
    *args: Any,
) -> None:
    """Check the computation of data profiles when the evaluations number is limited.

    Args:
        problem_configuration_class: The class of problem configuration.
        create_problem: A function that creates problems.
        target_values: The target values.
        algorithm_configurations: The algorithm configurations.
        results: The benchmarking results.
        *args: Positional arguments for the class initialization.
    """
    matplotlib.pyplot.close("all")
    target_values.compute_target_hits_history = mock.Mock(return_value=[0, 0, 0, 1])
    bench_problem = problem_configuration_class(
        "Problem", create_problem, *args, target_values=target_values
    )
    bench_problem.compute_data_profile(
        algorithm_configurations, results, max_iteration_number=4
    )


def check_variable_space(
    problem_configuration: BaseProblemConfiguration, variable_space: DesignSpace
) -> None:
    """Check the variable space.

    Args:
        problem_configuration: The problem configuration.
        variable_space: The variable space of the problem configuration.
    """
    assert problem_configuration.variable_space is variable_space


def check_worker_type(
    problem_configuration: BaseProblemConfiguration, worker_type: type[BaseWorker]
) -> None:
    """Check the type of benchmarking worker.

    Args:
        problem_configuration: The problem configuration.
        worker_type: The type of benchmarking worker.
    """
    assert problem_configuration.worker is worker_type


def check_number_of_scalar_constraints(
    problem_configuration: BaseProblemConfiguration, number_of_scalar_constraints: int
) -> None:
    """Check the number of scalar constraints.

    Args:
        problem_configuration: The problem configuration.
        number_of_scalar_constraints: The number of scalar constraints.
    """
    assert (
        problem_configuration.number_of_scalar_constraints
        == number_of_scalar_constraints
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
