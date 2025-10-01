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
"""Fixtures for the tests."""

from __future__ import annotations

import datetime
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from gemseo import create_mda
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.testing.pytest_conftest import *  # noqa: F401,F403
from numpy import array

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems.mda_problem_configuration import MDAProblemConfiguration
from gemseo_benchmark.problems.mdo_problem_configuration import MDOProblemConfiguration
from gemseo_benchmark.problems.optimization_problem_configuration import (
    OptimizationProblemConfiguration,
)
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.report.axis_data import IterationData
from gemseo_benchmark.results.performance_histories import PerformanceHistories
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo_benchmark.problems.mda_problem_configuration import MDAProblemType
    from gemseo_benchmark.problems.mdo_problem_configuration import MDOProblemType

design_variables = array([0.0, 1.0])


@pytest.fixture(scope="package")
def design_space() -> mock.Mock:
    """A design space."""
    design_space = mock.MagicMock()
    design_space.dimension = 2
    design_space.variable_names = ["x"]
    design_space.variable_sizes = {"x": 2}
    design_space.get_current_value = mock.Mock(return_value=design_variables)
    design_space.has_current_value = True
    design_space.set_current_value = mock.Mock()
    design_space.unnormalize_vect = lambda _: _
    design_space.untransform_vect = lambda x, no_check: x
    design_space.normalize = {"x": [True]}
    design_space.__iter__.return_value = ["x"]
    design_space.get_size = mock.Mock(return_value=2)
    return design_space


@pytest.fixture(scope="package")
def objective() -> mock.Mock:
    """An objective function."""
    objective = mock.Mock()
    objective.name = "f"
    return objective


@pytest.fixture(scope="package")
def inequality_constraint() -> mock.Mock:
    """An inequality constraint."""
    ineq_constr = mock.Mock()
    ineq_constr.name = "g"
    ineq_constr.f_type = "ineq"
    return ineq_constr


@pytest.fixture(scope="package")
def equality_constraint() -> mock.Mock:
    """An equality constraint."""
    eq_constr = mock.Mock()
    eq_constr.name = "h"
    eq_constr.f_type = "eq"
    return eq_constr


@pytest.fixture(scope="package")
def constraints(inequality_constraint, equality_constraint) -> mock.Mock:
    """A collection of constraints."""
    constraints = mock.Mock()
    constraints.get_names = mock.Mock(
        return_value=[inequality_constraint.name, equality_constraint.name]
    )
    constraints.get_number_of_unsatisfied_constraints = mock.Mock(return_value=1)
    return constraints


@pytest.fixture(scope="package")
def functions_values(
    objective, inequality_constraint, equality_constraint
) -> dict[str, float | ndarray]:
    """The values of the functions of a problem."""
    return {
        objective.name: 2.0,
        inequality_constraint.name: array([1.0]),
        equality_constraint.name: array([0.0]),
    }


@pytest.fixture(scope="package")
def hashable_array() -> mock.Mock:
    """A hashable array."""
    hashable_array = mock.Mock()
    hashable_array.unwrap = mock.Mock(return_value=design_variables)
    return hashable_array


@pytest.fixture(scope="package")
def database(hashable_array, functions_values) -> mock.Mock:
    """A database."""
    database = mock.Mock()
    database.items = mock.Mock(return_value=[(hashable_array, functions_values)])
    database.get = mock.Mock(return_value=functions_values)
    database.__len__ = mock.Mock(return_value=1)
    return database


@pytest.fixture(scope="package")
def minimization_problem(design_space, objective, constraints) -> mock.Mock:
    """A solved minimization problem."""
    problem = mock.Mock(spec=OptimizationProblem)
    problem.tolerances.inequality = 1e-4
    problem.tolerances.equality = 1e-2
    problem.design_space = design_space
    problem.design_space.dimension = design_space.dimension
    problem.objective = objective
    problem.minimize_objective = True
    problem.history = mock.Mock()
    problem.history.check_design_point_is_feasible = mock.Mock(
        return_value=(False, 1.0)
    )
    problem.constraints = constraints
    problem.scalar_constraint_names = constraints.get_names()
    return problem


@pytest.fixture(scope="package")
def maximization_problem(design_space, objective, constraints) -> mock.Mock:
    """A solved maximization problem."""
    problem = mock.Mock(spec=OptimizationProblem)
    problem.tolerances.inequality = 1e-4
    problem.tolerances.equality = 1e-2
    problem.design_space = design_space
    problem.design_space.dimension = design_space.dimension
    problem.objective = objective
    problem.minimize_objective = False
    problem.history = mock.Mock()
    problem.history.check_design_point_is_feasible = mock.Mock(
        return_value=(False, 1.0)
    )
    problem.constraints = constraints
    problem.scalar_constraint_names = constraints.get_names()
    return problem


def side_effect(
    algos_configurations,
    results,
    show=False,
    file_path=None,
    plot_all_histories=False,
    infeasibility_tolerance=0.0,
    max_eval_number=None,
    use_log_scale=False,
    plot_settings=READ_ONLY_EMPTY_DICT,
    grid_settings=READ_ONLY_EMPTY_DICT,
):
    """Side effect for the computation of a data profile."""
    shutil.copyfile(str(Path(__file__).parent / "data_profile.png"), str(file_path))


@pytest.fixture(scope="package")
def problem_a() -> mock.Mock:
    """A problem."""
    problem = mock.Mock()
    problem.name = "Problem A"
    problem.description = "The description of problem A."
    problem.optimum = 1.0
    problem.target_values = TargetValues([problem.optimum])
    problem.minimization_target_values = TargetValues([problem.optimum])
    problem.minimize_performance_measure = True
    problem.compute_data_profile = mock.Mock(side_effect=side_effect)
    problem.performance_measure_label = "Best feasible objective value"
    problem.number_of_scalar_constraints = 6
    problem.abscissa_data_type = IterationData
    return problem


@pytest.fixture(scope="package")
def problem_b() -> mock.Mock:
    """Another problem."""
    problem = mock.Mock()
    problem.name = "Problem B"
    problem.description = "The description of problem B."
    problem.optimum = None
    problem.target_values = TargetValues([-1.0])
    problem.minimization_target_values = TargetValues([1.0])
    problem.minimize_performance_measure = False
    problem.compute_data_profile = mock.Mock(side_effect=side_effect)
    problem.performance_measure_label = "Best feasible objective value"
    problem.number_of_scalar_constraints = 6
    problem.abscissa_data_type = IterationData
    return problem


@pytest.fixture(scope="package")
def group(problem_a, problem_b) -> mock.Mock:
    """The group of problems."""
    group = mock.MagicMock()
    group.name = "A group"
    group.description = "The description of the group."
    group.__iter__.return_value = [problem_a, problem_b]

    def side_effect(
        algos_configurations,
        histories_paths,
        show=False,
        plot_path=None,
        infeasibility_tolerance=0.0,
        max_eval_number=None,
        plot_settings=READ_ONLY_EMPTY_DICT,
        grid_settings=READ_ONLY_EMPTY_DICT,
        use_abscissa_log_scale=False,
    ):
        shutil.copyfile(str(Path(__file__).parent / "data_profile.png"), str(plot_path))

    group.compute_data_profile = mock.Mock(side_effect=side_effect)
    return group


@pytest.fixture(scope="package")
def algorithm_configuration() -> mock.Mock:
    """The configuration of an algorithm."""
    algo_config = mock.Mock()
    algo_config.algorithm_name = "SLSQP"
    algo_config.algorithm_options = {"normalize_design_space": False, "max_iter": 3}
    algo_config.name = "SLSQP"
    algo_config.instance_algorithm_options = {}
    algo_config.copy = mock.Mock(return_value=algo_config)
    algo_config.to_dict = mock.Mock(
        return_value={
            "configuration_name": "SLSQP",
            "algorithm_name": "SLSQP",
            "algorithm_options": {"normalize_design_space": False, "max_iter": 3},
        }
    )
    return algo_config


@pytest.fixture(scope="package")
def algorithms_configurations(algorithm_configuration) -> mock.Mock:
    """The configurations of algorithms."""
    algos_configs = mock.MagicMock()
    algos_configs.name = "algorithms configurations"
    algos_configs.names = [algorithm_configuration.name]
    algos_configs.algorithms = [algorithm_configuration.algorithm_name]
    algos_configs.__iter__.return_value = [algorithm_configuration]
    return algos_configs


@pytest.fixture(scope="package")
def unknown_algorithm_configuration():
    """The configuration of an algorithm unknown to GEMSEO."""
    algo_config = mock.Mock()
    algo_config.algorithm_name = "Algorithm"
    algo_config.algorithm_options = {}
    algo_config.name = "Configuration"
    algo_config.instance_algorithm_options = {}
    algo_config.copy = mock.Mock(return_value=algo_config)
    return algo_config


@pytest.fixture(scope="package")
def unknown_algorithms_configurations(
    algorithm_configuration, unknown_algorithm_configuration
) -> mock.Mock:
    """The configurations of algorithms unknown to GEMSEO."""
    algos_configs = mock.MagicMock()
    algos_configs.name = "unknown algorithms configurations"
    algos_configs.names = [
        algorithm_configuration.name,
        unknown_algorithm_configuration.name,
    ]
    algos_configs.algorithms = [
        algorithm_configuration.algorithm_name,
        unknown_algorithm_configuration.algorithm_name,
    ]
    algos_configs.__iter__.return_value = [
        algorithm_configuration,
        unknown_algorithm_configuration,
    ]
    return algos_configs


ALGO_NAME = "SLSQP"


@pytest.fixture
def results(
    algorithm_configuration, unknown_algorithm_configuration, problem_a, problem_b
) -> mock.Mock:
    """The results of the benchmarking."""
    results = mock.Mock()
    results.algorithms = [
        algorithm_configuration.name,
        unknown_algorithm_configuration.name,
    ]
    results.get_problems = mock.Mock(return_value=[problem_a.name, problem_b.name])
    directory = Path(__file__).parent
    paths = [
        directory / "history.json",
        directory / "unfeasible_history.json",
    ]
    results.get_paths = mock.Mock(return_value=paths)
    return results


@pytest.fixture(scope="package")
def rosenbrock() -> OptimizationProblemConfiguration:
    """A problem configuration based on the 2-dimensional Rosenbrock function."""
    return OptimizationProblemConfiguration(
        "Rosenbrock",
        Rosenbrock,
        [array([0.0, 1.0]), array([1.0, 0.0])],
        TargetValues([1e-2, 1e-4, 1e-6, 0.0]),
        optimum=0.0,
    )


@pytest.fixture(scope="package")
def problems_group(rosenbrock) -> ProblemsGroup:
    """A group of problems."""
    return ProblemsGroup("Rosenbrock", [rosenbrock])


@pytest.fixture(scope="module")
def results_root(tmp_path_factory) -> Path:
    """The root the L-BFGS-B results file tree."""
    return tmp_path_factory.mktemp("results")


@pytest.fixture(scope="package")
def performance_histories() -> PerformanceHistories:
    """A collection of performance histories."""
    return PerformanceHistories(
        PerformanceHistory([1.0, -1.0, 0.0], [2.0, 0.0, 3.0]),
        PerformanceHistory([-2.0, -2.0, 2.0], [0.0, 3.0, 0.0]),
        PerformanceHistory([3.0, -3.0, 3.0], [0.0, 0.0, 0.0]),
        PerformanceHistory([0.0, -2.0, 4.0], [0.0, 0.0, 0.0]),
    )


def mda_create_problem(
    algorithm_configuration: AlgorithmConfiguration,
) -> MDAProblemType:
    """Create an MDA problem."""
    disciplines = (
        AnalyticDiscipline({"y1": "x1 + y2"}),
        AnalyticDiscipline({"y2": "x2 - y1"}),
    )
    return create_mda(
        algorithm_configuration.algorithm_name,
        disciplines,
        **algorithm_configuration.algorithm_options,
    ), disciplines


@pytest.fixture(scope="module")
def multidisciplinary_variable_space() -> DesignSpace:
    """The variable space of a multidisciplinary problem configuration."""
    variable_space = DesignSpace()
    variable_space.add_variable("y1", lower_bound=0, upper_bound=1, value=0.5)
    variable_space.add_variable("y2", lower_bound=0, upper_bound=1, value=0.5)
    return variable_space


@pytest.fixture(scope="module")
def mda_problem_configuration(
    multidisciplinary_variable_space,
) -> MDAProblemConfiguration:
    """A problem configuration for multidisciplinary analysis."""
    return MDAProblemConfiguration(
        "Linear MDA",
        mda_create_problem,
        multidisciplinary_variable_space,
        starting_points=[array([0, 1]), array([1, 0])],
    )


@pytest.fixture(scope="module")
def mda_algorithm_configuration() -> AlgorithmConfiguration:
    """An algorithm configuration for multidisciplinary analysis."""
    return AlgorithmConfiguration("MDAJacobi")


def mdo_create_problem(
    algorithm_configuration: AlgorithmConfiguration,
) -> MDOProblemType:
    """Create an MDO problem."""
    disciplines = (
        AnalyticDiscipline({"y1": "x1 + y2"}),
        AnalyticDiscipline({"y2": "x2 - y1"}),
    )
    variable_space = DesignSpace()
    variable_space.add_variable("x1", lower_bound=0, upper_bound=10, value=5)
    variable_space.add_variable("x2", lower_bound=0, upper_bound=10, value=5)
    scenario = create_scenario(
        disciplines, "y1", variable_space, formulation_name="MDF"
    )
    scenario.set_algorithm(
        algo_name=algorithm_configuration.algorithm_name,
        **algorithm_configuration.algorithm_options,
    )
    return scenario, disciplines


@pytest.fixture(scope="module")
def mdo_problem_configuration(
    multidisciplinary_variable_space,
) -> MDOProblemConfiguration:
    """A problem configuration for multidisciplinary optimization."""
    return MDOProblemConfiguration(
        "Linear MDO",
        mdo_create_problem,
        multidisciplinary_variable_space,
        True,
        0,
        starting_points=[array([0, 1]), array([1, 0])],
    )


@pytest.fixture(scope="module")
def mdo_algorithm_configuration() -> AlgorithmConfiguration:
    """An algorithm configuration for multidisciplinary optimization."""
    return AlgorithmConfiguration("SLSQP")


@pytest.fixture(scope="module")
def timed_performance_histories() -> PerformanceHistories:
    """Performance histories with given elapsed times."""
    return PerformanceHistories(
        PerformanceHistory(
            [1, 1],
            elapsed_times=[
                datetime.timedelta(seconds=1),
                datetime.timedelta(seconds=3),
            ],
        ),
        PerformanceHistory(
            [2, 2],
            elapsed_times=[
                datetime.timedelta(seconds=2),
                datetime.timedelta(seconds=4),
            ],
        ),
    )


def check_spread_over_time(histories: PerformanceHistories) -> None:
    """Check the spreading of performance histories over a timeline.

    Args:
        histories: The performance histories.
    """
    timeline = [datetime.timedelta(seconds=i) for i in range(1, 5)]
    assert len(histories) == 2

    assert len(histories[0]) == 4
    assert [item.performance_measure for item in histories[0]] == [1] * 4
    assert [item.n_unsatisfied_constraints for item in histories[0]] == [0] * 4
    assert [item.elapsed_time for item in histories[0]] == timeline

    assert len(histories[1]) == 4
    assert math.isnan(histories[1][0].performance_measure)
    assert histories[1][0].n_unsatisfied_constraints == 5
    assert [item.performance_measure for item in histories[1][1:]] == [2] * 3
    assert [item.n_unsatisfied_constraints for item in histories[1][1:]] == [0] * 3
    assert [item.elapsed_time for item in histories[1]] == timeline


@pytest.fixture(scope="module")
def multidisciplinary_histories() -> PerformanceHistories:
    """Performance histories with given numbers of discipline executions."""
    return PerformanceHistories(
        PerformanceHistory([1, 1], number_of_discipline_executions=[1, 3]),
        PerformanceHistory([2, 2], number_of_discipline_executions=[2, 4]),
    )


def check_spread_over_numbers_of_discipline_executions(
    histories: PerformanceHistories,
) -> None:
    """Check the spreading of performance histories over numbers of executions.

    Args:
        histories: The performance histories.
    """
    numbers_of_discipline_executions = [1, 2, 3, 4]
    assert len(histories) == 2

    assert len(histories[0]) == 4
    assert [item.performance_measure for item in histories[0]] == [1] * 4
    assert [item.n_unsatisfied_constraints for item in histories[0]] == [0] * 4
    assert [
        item.number_of_discipline_executions for item in histories[0]
    ] == numbers_of_discipline_executions

    assert len(histories[1]) == 4
    assert math.isnan(histories[1][0].performance_measure)
    assert histories[1][0].n_unsatisfied_constraints == 5
    assert [item.performance_measure for item in histories[1][1:]] == [2] * 3
    assert [item.n_unsatisfied_constraints for item in histories[1][1:]] == [0] * 3
    assert [
        item.number_of_discipline_executions for item in histories[1]
    ] == numbers_of_discipline_executions
