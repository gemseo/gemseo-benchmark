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
"""The interface for benchmarking workers.

A benchmarking worker is responsible for:

1. the execution of algorithm configurations on problem configurations,
2. the creation of the associated performance histories.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from gemseo import LOGGER as GEMSEO_LOGGER
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from gemseo.algos.base_algo_factory import BaseAlgoFactory
    from gemseo.typing import RealArray

    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )
    from gemseo_benchmark.benchmarker._metrics import BaseMetrics
    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )
    from gemseo_benchmark.results.performance_history import PerformanceHistory


ProblemType = Any
"""The type of problem."""


class BaseWorker(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for benchmarking workers."""

    @property
    @abstractmethod
    def _algorithm_factory() -> BaseAlgoFactory:
        """The algorithm factory."""

    @classmethod
    def check_algorithm_availability(cls, algorithm_name: str) -> None:
        """Check whether an algorithm is available.

        Args:
            algorithm_name: The name of the algorithm.

        Raises:
            ValueError: If the algorithm is not available.
        """
        if not cls._algorithm_factory.is_available(algorithm_name):
            msg = f"The algorithm {algorithm_name!r} is not available."
            raise ValueError(msg)

    @classmethod
    def execute(
        cls,
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: BaseProblemConfiguration,
        starting_point: RealArray,
        gemseo_log_message: str,
        log_path: Path | None,
        performance_history_path: Path,
        hdf_file_path: Path | None,
        benchmarking_logger: logging.Logger,
    ) -> None:
        """Create a performance history from a problem.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem_configuration: The problem configuration.
            starting_point: The starting point of the algorithm.
            gemseo_log_message: The message to log before benchmarking.
            log_path: The file path to save the log.
                If ``None``, the log is not saved.
            performance_history_path: The file path to save the performance history.
            hdf_file_path: The HDF file path.
                If ``None``, no HDF file will be written.
            benchmarking_logger: The benchmarking logger.
        """
        # Start writing in the log file.
        if log_path is not None:
            file_handler = logging.FileHandler(log_path, "w")
            loggers = [benchmarking_logger, GEMSEO_LOGGER]
            for logger in loggers:
                logger.addHandler(file_handler)

        problem = cls._get_problem(
            algorithm_configuration,
            problem_configuration,
            starting_point,
            hdf_file_path,
        )

        metrics_listeners = cls._add_metrics_listeners(problem)

        benchmarking_logger.info(gemseo_log_message)
        with Timer() as timer:
            cls._execute(
                algorithm_configuration,
                problem_configuration,
                starting_point,
                problem,
            )

        # Stop writing in the log file.
        if log_path is not None:
            for logger in loggers:
                logger.removeHandler(file_handler)

            file_handler.close()

        cls._create_performance_history(
            algorithm_configuration,
            problem_configuration,
            problem,
            timer,
            metrics_listeners,
        ).to_file(performance_history_path)
        cls._post_execute(problem, hdf_file_path)

    @staticmethod
    @abstractmethod
    def _get_problem(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: BaseProblemConfiguration,
        starting_point: RealArray,
        hdf_file_path: Path | None,
    ) -> ProblemType:
        """Return a problem ready for execution.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem_configuration: The problem configuration.
            starting_point: The starting point of the algorithm.
            hdf_file_path: The HDF file path.
                If ``None``, no HDF file will be written.

        Return:
            A problem ready for execution.
        """

    @classmethod
    @abstractmethod
    def _add_metrics_listeners(cls, problem: ProblemType) -> tuple[BaseMetrics, ...]:
        """Add the listeners for the metrics of an execution.

        Args:
            problem: A problem.

        Returns:
            The metrics listeners.
        """

    @staticmethod
    @abstractmethod
    def _execute(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: BaseProblemConfiguration,
        problem: ProblemType,
        starting_point: RealArray,
    ) -> None:
        """Execute an algorithm on a problem configuration from a starting point.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem_configuration: The problem configuration.
            problem: A problem.
            starting_point: The starting point of the algorithm.
        """

    @staticmethod
    @abstractmethod
    def _create_performance_history(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: BaseProblemConfiguration,
        problem: ProblemType,
        timer: Timer,
        metrics_listeners: Iterable[BaseMetrics],
    ) -> PerformanceHistory:
        """Create a performance history from a solved problem.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem_configuration: The problem configuration.
            problem: A problem.
            timer: The timer of the worker execution.
            metrics_listeners: The metrics listeners.

        Return:
            The performance history.
        """

    @staticmethod
    @abstractmethod
    def _post_execute(problem: ProblemType, hdf_file_path: Path | None) -> None:
        """Run instructions after the execution of the worker.

        Args:
            problem: A problem.
            hdf_file_path: The HDF file path.
                If ``None``, no HDF file will be written.
        """
