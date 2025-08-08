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

"""Tests for the multidisciplinary optimization benchmarking worker."""

from pathlib import Path

import numpy
import pytest
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.testing.pytest_conftest import tmp_wd  # noqa: F401
from gemseo.utils.timer import Timer

from gemseo_benchmark.benchmarker.mdo_worker import MDOWorker
from tests.benchmarker.conftest import check_algorithm_availability
from tests.benchmarker.conftest import check_algorithm_factory
from tests.benchmarker.conftest import check_create_performance_history
from tests.benchmarker.conftest import check_execution


def test_algorithm_factory() -> None:
    """Check the algorithm factory."""
    check_algorithm_factory(MDOWorker, OptimizationLibraryFactory)


def test_check_algorithm_availability() -> None:
    """Check the algorithm availability checker."""
    check_algorithm_availability(MDOWorker)


@pytest.mark.parametrize("save_gemseo_log", [False, True])
@pytest.mark.parametrize("save_data", [False, True])
def test_execution(
    tmp_wd,  # noqa: F811
    algorithm_configuration,
    mdo_problem_configuration,
    save_gemseo_log,
    save_data,
) -> None:
    """Check the execution of the benchmarking worker."""
    check_execution(
        tmp_wd,
        algorithm_configuration,
        mdo_problem_configuration,
        save_gemseo_log,
        save_data,
        MDOWorker,
        "",
    )


@pytest.mark.parametrize("save_data", [False, True])
def test_get_problem(
    tmp_wd,  # noqa: F811
    algorithm_configuration,
    mdo_problem_configuration,
    save_data,
) -> None:
    """Check the problem getter."""
    hdf_file_path = Path("data.h5") if save_data else None
    scenario = MDOWorker._get_problem(
        algorithm_configuration,
        mdo_problem_configuration,
        numpy.zeros(2),
        hdf_file_path,
    )[0]
    assert isinstance(scenario, MDOScenario)


def test_execute(
    algorithm_configuration, mdo_problem_configuration, enable_discipline_statistics
) -> None:
    """Check the execution of the benchmarking worker."""
    problem = mdo_problem_configuration.create_problem(algorithm_configuration)
    scenario, (discipline1, discipline2) = problem
    assert scenario.execution_statistics.n_executions == 0
    assert discipline1.execution_statistics.n_executions == 0
    assert discipline2.execution_statistics.n_executions == 0
    MDOWorker._execute(
        algorithm_configuration,
        mdo_problem_configuration,
        mdo_problem_configuration.variable_space.get_lower_bounds(),
        problem,
    )
    assert scenario.execution_statistics.n_executions == 1
    assert discipline1.execution_statistics.n_executions > 0
    assert discipline2.execution_statistics.n_executions > 0


def test_create_performance_history(
    algorithm_configuration, mdo_problem_configuration
) -> None:
    """Check the creation of the performance history."""
    problem = mdo_problem_configuration.create_problem(algorithm_configuration)
    scenario = problem[0]
    with Timer() as timer:
        scenario.execute()

    optimization_problem = scenario.formulation.optimization_problem
    database = optimization_problem.database
    database_size = len(database)
    check_create_performance_history(
        algorithm_configuration,
        mdo_problem_configuration,
        problem,
        MDOWorker,
        timer,
        database.get_function_history(optimization_problem.objective_name),
        [0.0] * database_size,
        [0] * database_size,
    )
