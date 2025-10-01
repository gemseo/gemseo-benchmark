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
"""Benchmarking algorithm configurations on problem configurations."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import TYPE_CHECKING

from gemseo import LOGGER as GEMSEO_LOGGER

from gemseo_benchmark import join_substrings
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.results.results import Results

if TYPE_CHECKING:
    from collections.abc import Iterable
    from concurrent.futures import Executor
    from concurrent.futures import Future
    from pathlib import Path

    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.benchmarker.base_worker import BaseWorker
    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )

LOGGER = logging.getLogger(__name__)


class Benchmarker:
    """A class to benchmark algorithm configurations on problem configurations."""

    __hdf_path: Path | None
    """The path to the destination directory for the HDF files if saved."""

    __histories_path: Path
    """The path to the directory where to save the performance histories."""

    _results: Results
    """A collection of paths to performance histories."""

    __results_path: Path | None
    """The path to the file for saving the performance histories paths."""

    def __init__(
        self,
        histories_path: Path,
        results_path: Path | None = None,
        hdf_path: Path | None = None,
    ) -> None:
        """
        Args:
            histories_path: The path to the directory where to save the performance
                histories.
            results_path: The path to the file for saving the performance histories
                paths.
                If exists, the file is updated with the new performance histories paths.
                If ``None``, no performance history path will be saved.
            hdf_path: The path to the destination directory for the HDF files.
                If ``None``, no HDF file will be saved.
        """  # noqa: D205, D212, D415
        self.__hdf_path = hdf_path
        self.__histories_path = histories_path
        self.__results_path = results_path
        if results_path is not None and results_path.is_file():
            self._results = Results(results_path)
        else:
            self._results = Results()

    def execute(
        self,
        problem_configurations: Iterable[BaseProblemConfiguration],
        algorithm_configurations: AlgorithmsConfigurations,
        overwrite_histories: bool = False,
        n_processes: int = 1,
        use_threading: bool = False,
        save_log: bool = False,
    ) -> Results:
        """Execute algorithm configurations on problem configurations.

        Args:
            problem_configurations: The problem configurations.
            algorithm_configurations: The algorithms configurations.
            overwrite_histories: Whether to overwrite the existing performance
                histories.
            n_processes: The maximum simultaneous number of threads or processes
                used to parallelize the execution.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution.
            save_log: Whether to save the log to a file.
                If ``use_threading`` is ``True``, a single global file will be saved
                in the performance histories directory.
                Otherwise, one file per optimization will be saved
                next to each performance history file.

        Returns:
            The results of the benchmarking.
        """
        if save_log and use_threading:
            # Set one file handler for all threads.
            file_handler = logging.FileHandler(
                self.__histories_path / "gemseo.log", "w"
            )
            file_handler.setFormatter(logging.Formatter("%(threadName)s %(message)s"))
            loggers = [LOGGER, GEMSEO_LOGGER]
            for logger in loggers:
                logger.addHandler(file_handler)

        executor_class = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
        with executor_class(max_workers=n_processes) as executor:
            future_to_path = {}
            for original_algorithm_configuration in algorithm_configurations:
                algorithm_configuration = original_algorithm_configuration.copy()
                for problem_configuration in problem_configurations:
                    worker = problem_configuration.worker
                    worker.check_algorithm_availability(
                        algorithm_configuration.algorithm_name
                    )
                    future_to_path.update(
                        self.__execute(
                            executor,
                            worker,
                            algorithm_configuration,
                            problem_configuration,
                            overwrite_histories,
                            save_log,
                            use_threading,
                        )
                    )

        for future in as_completed(future_to_path):
            exception = future.exception()
            if exception is None:
                self._results.add_path(*future_to_path[future][1:])
            else:
                LOGGER.warning(
                    "%s raised: %s", future_to_path[future][0][:-1], exception
                )

        if save_log and use_threading:
            for logger in loggers:
                logger.removeHandler(file_handler)

            file_handler.close()

        if future_to_path and self.__results_path:
            self._results.to_file(self.__results_path, 4)

        return self._results

    def __execute(
        self,
        executor: Executor,
        worker: BaseWorker,
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: BaseProblemConfiguration,
        overwrite_histories: bool,
        save_log: bool,
        use_threading: bool,
    ) -> dict[Future, tuple[str, str, str, Path]]:
        """Execute an algorithm configuration on a problem configuration.

        Args:
            executor: The executor for parallel execution.
            worker: The worker of the problem configuration.
            algorithm_configuration: The algorithm configuration.
            problem_configuration: The problem configuration.
            overwrite_histories: Whether to overwrite the existing performance
                histories.
            save_log: Whether to save the log to a file
                next to the performance history file.
            use_threading: Whether threads are used instead of processes
                to parallelize the execution.

        Returns:
            The executions.
        """
        future_to_path = {}
        algorithm_configuration_name = algorithm_configuration.name
        problem_configuration_name = problem_configuration.name
        if overwrite_histories:
            self._results.remove_paths(
                algorithm_configuration_name, problem_configuration_name
            )

        for starting_point_index, starting_point in enumerate(
            problem_configuration.starting_points
        ):
            gemseo_log_message = self.__is_problem_unsolved(
                algorithm_configuration,
                problem_configuration,
                starting_point_index,
                overwrite_histories,
            )
            if not gemseo_log_message:
                continue

            performance_history_path = self.get_history_path(
                algorithm_configuration,
                problem_configuration_name,
                starting_point_index,
                True,
            )

            if save_log and not use_threading:
                gemseo_log_path = performance_history_path.with_suffix(".log")
            else:
                gemseo_log_path = None

            if self.__hdf_path is not None:
                hdf_file_path = self._get_path(
                    self.__hdf_path,
                    algorithm_configuration,
                    problem_configuration_name,
                    starting_point_index,
                    "h5",
                    True,
                )
            else:
                hdf_file_path = None

            future_to_path[
                executor.submit(
                    worker.execute,
                    self.__set_problem_algorithm_options(
                        algorithm_configuration,
                        problem_configuration,
                        starting_point_index,
                    ),
                    problem_configuration,
                    starting_point,
                    gemseo_log_message,
                    gemseo_log_path,
                    performance_history_path,
                    hdf_file_path,
                    LOGGER,
                )
            ] = (
                gemseo_log_message,
                algorithm_configuration_name,
                problem_configuration_name,
                performance_history_path,
            )

        return future_to_path

    def __is_problem_unsolved(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: BaseProblemConfiguration,
        index: int,
        overwrite_histories: bool,
    ) -> str:
        """Check whether a problem needs to be solved.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem_configuration: The problem configuration.
            index: The index of the problem.
            overwrite_histories: Whether to overwrite existing histories.

        Returns:
            The message to log when solving the problem.
        """
        problem_configuration_name = problem_configuration.name
        execution_info = (
            f"problem {index + 1} "
            f"of problem configuration {problem_configuration_name} "
            f"for algorithm configuration {algorithm_configuration.name}"
        )

        if overwrite_histories or not self._results.contains(
            algorithm_configuration.name,
            problem_configuration_name,
            self.get_history_path(
                algorithm_configuration, problem_configuration_name, index
            ),
        ):
            return f"Solving {execution_info}."

        LOGGER.info("Skipping %s.", execution_info)
        return ""

    @staticmethod
    def __set_problem_algorithm_options(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: BaseProblemConfiguration,
        index: int,
    ) -> AlgorithmConfiguration:
        """Return the algorithm configuration of a problem.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem_configuration: The problem configuration.
            index: The 0-based index of the problem.

        Returns:
            The algorithm configuration of the problem.
        """
        algorithm_options = dict(algorithm_configuration.algorithm_options)
        for name, value in algorithm_configuration.instance_algorithm_options.items():
            algorithm_options[name] = value(problem_configuration, index)

        return AlgorithmConfiguration(
            algorithm_configuration.algorithm_name,
            algorithm_configuration.name,
            {},
            **algorithm_options,
        )

    def get_history_path(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration_name: str,
        index: int,
        make_parents: bool = False,
    ) -> Path:
        """Return a path for a history file.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem_configuration_name: The name of the problem configuration.
            index: The index of the problem.
            make_parents: Whether to make the parent directories.

        Returns:
            The path for the history file.
        """
        return self._get_path(
            self.__histories_path,
            algorithm_configuration,
            problem_configuration_name,
            index,
            "json",
            make_parents=make_parents,
        )

    @staticmethod
    def _get_path(
        root_dir: Path,
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration_name: str,
        index: int,
        extension: str = "json",
        make_parents: bool = False,
    ) -> Path:
        """Return a path in the file tree dedicated to a specific optimization run.

        Args:
            root_dir: The path to the root directory.
            algorithm_configuration: The algorithm configuration.
            problem_configuration_name: The name of the problem configuration.
            index: The index of the problem.
            extension: The extension of the path.
                If ``None``, the extension is for a JSON file.
            make_parents: Whether to make the parent directories of the path.

        Returns:
            The path for the file.
        """
        configuration_name = join_substrings(algorithm_configuration.name)
        file_path = (
            root_dir.resolve()
            / configuration_name
            / join_substrings(problem_configuration_name)
            / f"{configuration_name}.{index + 1}.{extension}"
        )
        if make_parents:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        return file_path
