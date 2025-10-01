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

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from numpy.testing import assert_equal

from gemseo_benchmark.problems.optimization_problem_configuration import (
    OptimizationProblemConfiguration,
)

if TYPE_CHECKING:
    from gemseo.core.base_factory import BaseFactory
    from gemseo.utils.timer import Timer

    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )
    from gemseo_benchmark.benchmarker.base_worker import BaseWorker
    from gemseo_benchmark.benchmarker.base_worker import ProblemType
    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )


def check_algorithm_factory(
    worker: BaseWorker, factory_type: type[BaseFactory]
) -> None:
    """Check the algorithm factory.

    Args:
        worker: The benchmarking worker.
        factory_type: The type of factory.
    """
    assert isinstance(worker._algorithm_factory, factory_type)


def check_algorithm_availability(worker: BaseWorker) -> None:
    """Check the algorithm availability checker.

    Args:
        worker: The benchmarking worker.
    """
    with pytest.raises(
        ValueError, match=r"The algorithm 'Algorithm' is not available."
    ):
        worker.check_algorithm_availability("Algorithm")


def check_execution(
    tmp_wd,  # noqa: F811
    algorithm_configuration: AlgorithmConfiguration,
    problem_configuration: BaseProblemConfiguration,
    save_gemseo_log: bool,
    save_data: bool,
    worker_type: type[BaseWorker],
    gemseo_log_message: str,
) -> None:
    """Check the execution of the benchmarking worker.

    Args:
        tmp_wd: The path to a temporary working directory.
        algorithm_configuration: The algorithm configuration.
        problem_configuration: The problem configuration.
        save_gemseo_log: Whether to save the GEMSEO log to file.
        save_data: Whether to save data to file.
        worker_type: The type of benchmarking worker.
        gemseo_log_message: The expected GEMSEO log message.
    """
    gemseo_benchmark_log_message = (
        f"Solving problem 1 of problem configuration {problem_configuration.name} "
        f"for algorithm configuration {algorithm_configuration.name}."
    )
    log_path = Path("gemseo.log") if save_gemseo_log else None
    performance_history_path = Path("performance_history.json")
    hdf_file_path = Path("data.h5") if save_data else None
    worker_type().execute(
        algorithm_configuration,
        problem_configuration,
        problem_configuration.variable_space.get_lower_bounds(),
        gemseo_benchmark_log_message,
        log_path,
        performance_history_path,
        hdf_file_path,
        logging.getLogger(),
    )
    if save_gemseo_log:
        with log_path.open("r") as file:
            log = file.read()
            assert gemseo_benchmark_log_message in log
            assert gemseo_log_message in log

    assert performance_history_path.is_file()
    if save_data:
        assert hdf_file_path.is_file()


def check_create_performance_history(
    algorithm_configuration: AlgorithmConfiguration,
    problem_configuration: BaseProblemConfiguration,
    problem: ProblemType,
    worker_type: type[BaseWorker],
    timer: Timer,
    performance_measures: list[float],
    infeasibility_measures: list[float],
    number_of_unsatisfied_constraints: list[int],
) -> None:
    """Check the creation of a performance history by a benchmarking worker.

    Args:
        algorithm_configuration: The algorithm configuration.
        problem_configuration: The problem configuration.
        problem: The problem from which to create a performance history.
        worker_type: The type of benchmarking worker.
        timer: The timer of the worker execution.
        performance_measures: The history of the performance measure.
        infeasibility_measures: The history of the infeasibility measure.
        n_unsatisfied_constraints: The history of the number of unsatisfied constraints.
    """
    elapsed_times = [
        datetime.timedelta(seconds=i + 1) for i in range(len(performance_measures))
    ]
    time_listener = mock.Mock()
    time_listener.get_metrics = mock.Mock(return_value=elapsed_times)
    metrics_listeners = [time_listener]
    if not isinstance(problem_configuration, OptimizationProblemConfiguration):
        discipline_listener = mock.Mock()
        discipline_listener.get_metrics = mock.Mock(
            return_value=range(1, len(performance_measures) + 1)
        )
        metrics_listeners.append(discipline_listener)

    performance_history = worker_type()._create_performance_history(
        algorithm_configuration,
        problem_configuration,
        problem,
        timer,
        metrics_listeners,
    )
    assert performance_history.algorithm_configuration is algorithm_configuration
    assert performance_history.problem_name == problem_configuration.name
    assert len(performance_history) == len(performance_measures)
    assert_equal(performance_history.performance_measures, performance_measures)
    assert performance_history.infeasibility_measures == infeasibility_measures
    assert (
        performance_history.n_unsatisfied_constraints
        == number_of_unsatisfied_constraints
    )
    assert performance_history.total_time == timer.elapsed_time
    assert [
        history_item.elapsed_time.total_seconds()
        for history_item in performance_history
    ] == list(range(1, len(performance_measures) + 1))
