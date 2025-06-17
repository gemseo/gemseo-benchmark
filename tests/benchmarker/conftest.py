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

import logging
from pathlib import Path

import pytest
from gemseo.core.base_factory import BaseFactory

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.benchmarker.base_worker import BaseWorker
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
