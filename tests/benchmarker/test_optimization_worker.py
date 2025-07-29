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
"""Tests for the benchmarking worker."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.utils.testing.pytest_conftest import tmp_wd  # noqa: F401
from gemseo.utils.timer import Timer
from numpy.testing import assert_equal

from gemseo_benchmark.benchmarker.optimization_worker import OptimizationWorker
from tests.benchmarker.conftest import check_create_performance_history


def test_get_algorithm_factory() -> None:
    """Check the algorithm factory."""
    assert isinstance(OptimizationWorker._algorithm_factory, OptimizationLibraryFactory)


def test_check_algorithm_availability() -> None:
    """Check the algorithm availability checker."""
    with pytest.raises(
        ValueError, match=r"The algorithm 'Algorithm' is not available."
    ):
        OptimizationWorker.check_algorithm_availability("Algorithm")


@pytest.mark.parametrize("save_gemseo_log", [False, True])
@pytest.mark.parametrize("save_data", [False, True])
def test_execution(
    tmp_wd,  # noqa: F811
    algorithm_configuration,
    rosenbrock,
    save_gemseo_log,
    save_data,
) -> None:
    """Check the execution of the benchmarking worker."""
    gemseo_log_message = (
        f"Solving problem 1 of problem configuration {rosenbrock.name} "
        f"for algorithm configuration {algorithm_configuration.name}."
    )
    gemseo_log_path = Path("gemseo.log") if save_gemseo_log else None
    performance_history_path = Path("performance_history.json")
    hdf_file_path = Path("database.h5") if save_data else None
    OptimizationWorker().execute(
        algorithm_configuration,
        rosenbrock,
        rosenbrock.create_problem().design_space.get_lower_bounds(),
        gemseo_log_message,
        gemseo_log_path,
        performance_history_path,
        hdf_file_path,
        logging.getLogger(),
    )
    if save_gemseo_log:
        with gemseo_log_path.open("r") as file:
            log = file.read()
            assert gemseo_log_message in log
            for line in [
                "Optimization problem:",
                "   minimize rosen(x) = "
                "sum( 100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2 )",
                "   with respect to x",
                "   over the design space:",
                "      +------+-------------+-------+-------------+-------+",
                "      | Name | Lower bound | Value | Upper bound | Type  |",
                "      +------+-------------+-------+-------------+-------+",
                "      | x[0] |      -2     |   -2  |      2      | float |",
                "      | x[1] |      -2     |   -2  |      2      | float |",
                "      +------+-------------+-------+-------------+-------+",
                "Solving optimization problem with algorithm SLSQP:",
            ]:
                assert line in log

    assert performance_history_path.is_file()
    if save_data:
        assert hdf_file_path.is_file()


def test_get_problem(algorithm_configuration, rosenbrock) -> None:
    """Check the problem getter."""
    starting_point = rosenbrock.create_problem().design_space.get_lower_bounds()
    problem = OptimizationWorker()._get_problem(
        algorithm_configuration, rosenbrock, starting_point, None
    )
    assert isinstance(problem, OptimizationProblem)
    assert_equal(problem.design_space.get_current_value(), starting_point)


def test_execute(algorithm_configuration, rosenbrock) -> None:
    """Check the execution of the benchmarking worker."""
    problem = rosenbrock.create_problem()
    starting_point = problem.design_space.get_lower_bounds()
    assert problem.solution is None
    OptimizationWorker()._execute(
        algorithm_configuration, rosenbrock, starting_point, problem
    )
    assert problem.solution is not None


def test_create_performance_history(algorithm_configuration, rosenbrock) -> None:
    """Check the creation of the performance history."""
    problem = rosenbrock.create_problem()
    problem.preprocess_functions()
    with Timer() as timer:
        values, _ = problem.evaluate_functions()

    check_create_performance_history(
        algorithm_configuration,
        rosenbrock,
        problem,
        OptimizationWorker,
        timer,
        [values[problem.standardized_objective_name]],
        [0.0],
        [0],
    )
