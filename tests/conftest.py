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

import shutil
from pathlib import Path
from unittest import mock

import pytest
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems.problem import Problem
from numpy import array
from numpy import ndarray

design_variables = array([0.0, 1.0])


@pytest.fixture(scope="package")
def design_space() -> mock.Mock:
    """A design space."""
    design_space = mock.Mock()
    design_space.dimension = 2
    design_space.variable_names = ["x"]
    design_space.variable_sizes = {"x": 2}
    design_space.get_current_value = mock.Mock(return_value=design_variables)
    design_space.set_current_value = mock.Mock()
    design_space.unnormalize_vect = lambda _: _
    design_space.untransform_vect = lambda x, no_check: x
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
def problem(
    design_space,
    objective,
    inequality_constraint,
    equality_constraint,
    functions_values,
) -> mock.Mock:
    """A solved optimization problem."""
    problem = mock.Mock(spec=OptimizationProblem)
    problem.ineq_tolerance = 1e-4
    problem.eq_tolerance = 1e-2
    problem.design_space = design_space
    problem.dimension = design_space.dimension
    problem.objective = objective
    problem.nonproc_objective = None
    problem.constraints = [inequality_constraint, equality_constraint]
    problem.get_constraint_names = mock.Mock(
        return_value=[inequality_constraint.name, equality_constraint.name]
    )
    problem.get_scalar_constraint_names = problem.get_constraint_names
    problem.evaluate_functions = mock.Mock(return_value=(functions_values, None))
    problem.get_violation_criteria = mock.Mock(return_value=(False, 1.0))
    problem.get_number_of_unsatisfied_constraints = mock.Mock(return_value=1)
    problem.get_optimum = mock.Mock(
        return_value=(
            functions_values[objective.name],
            design_variables,
            True,
            functions_values,
            None,
        )
    )
    problem.minimize_objective = True
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
    problem.compute_data_profile = mock.Mock(side_effect=side_effect)
    problem.plot_histories = mock.Mock(side_effect=side_effect)
    return problem


@pytest.fixture(scope="package")
def problem_b() -> mock.Mock:
    """Another problem."""
    problem = mock.Mock()
    problem.name = "Problem B"
    problem.description = "The description of problem B."
    problem.optimum = 2.0
    problem.target_values = TargetValues([problem.optimum])
    problem.compute_data_profile = mock.Mock(side_effect=side_effect)
    problem.plot_histories = mock.Mock(side_effect=side_effect)
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
    ):
        shutil.copyfile(str(Path(__file__).parent / "data_profile.png"), str(plot_path))

    group.compute_data_profile = mock.Mock(side_effect=side_effect)
    return group


@pytest.fixture(scope="package")
def algorithm_configuration() -> mock.Mock():
    """The configuration of an algorithm."""
    algo_config = mock.Mock()
    algo_config.algorithm_name = "SLSQP"
    algo_config.algorithm_options = {"normalize_design_space": False, "max_iter": 3}
    algo_config.name = "SLSQP"
    return algo_config


@pytest.fixture(scope="package")
def algorithms_configurations(algorithm_configuration) -> mock.Mock():
    """The configurations of algorithms."""
    algos_configs = mock.MagicMock()
    algos_configs.name = "algorithms configurations"
    algos_configs.names = [algorithm_configuration.name]
    algos_configs.algorithms = [algorithm_configuration.algorithm]
    algos_configs.__iter__.return_value = [algorithm_configuration]
    return algos_configs


@pytest.fixture(scope="package")
def unknown_algorithm_configuration():
    """The configuration of an algorithm unknown to GEMSEO."""
    algo_config = mock.Mock()
    algo_config.algorithm_name = "Algorithm"
    algo_config.algorithm_options = dict()
    algo_config.name = "Configuration"
    return algo_config


@pytest.fixture(scope="package")
def unknown_algorithms_configurations(
    algorithm_configuration, unknown_algorithm_configuration
) -> mock.Mock():
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


@pytest.fixture(scope="function")
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
    paths = [Path(__file__).parent / "history.json"]
    results.get_paths = mock.Mock(return_value=paths)
    return results


@pytest.fixture(scope="module")
def rosenbrock() -> Problem:
    """A benchmarking problem based on the 2-dimensional Rosenbrock function."""
    return Problem(
        "Rosenbrock",
        Rosenbrock,
        [array([0.0, 1.0]), array([1.0, 0.0])],
        TargetValues([1e-2, 1e-4, 1e-6, 0.0]),
        optimum=0.0,
    )


@pytest.fixture(scope="module")
def results_root(tmp_path_factory) -> Path:
    """The root the L-BFGS-B results file tree."""
    return tmp_path_factory.mktemp("results")
