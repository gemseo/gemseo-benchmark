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
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

from gemseo import logger as gemseo_logger

from gemseo_benchmark import join_substrings
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.benchmarker.benchmarking_error import BenchmarkingError
from gemseo_benchmark.results.results import Results

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from concurrent.futures import Executor
    from concurrent.futures import Future
    from pathlib import Path

    from gemseo.util.pydantic import BaseSettings

    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )

LOGGER = logging.getLogger(__name__)


class _Execution(NamedTuple):
    """An execution of an algorithm configuration on a problem."""

    algorithm_configuration_name: str
    """The name of the algorithm configuration."""

    problem_configuration: BaseProblemConfiguration
    """The problem configuration."""

    index: int
    """The 0-based index of the problem."""

    settings: BaseSettings
    """The settings of the algorithm for the problem."""

    message: str
    """The message to log when solving the problem."""


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
                If `None`, no performance history path will be saved.
            hdf_path: The path to the destination directory for the HDF files.
                If `None`, no HDF file will be saved.
        """  # ruff: ignore[missing-blank-line-after-summary, multi-line-summary-first-line, missing-terminal-punctuation]
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
        raise_errors: bool = False,
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
                If `use_threading` is `True`, a single global file will be saved
                in the performance histories directory.
                Otherwise, one file per optimization will be saved
                next to each performance history file.
            raise_errors: Whether to raise a `BenchmarkingError`
                when some executions raised exceptions,
                once all the executions are over
                and their performance histories are saved.
                If `False`, the failures are only logged,
                at the `ERROR` level, with their traceback,
                followed by a summary.

        Returns:
            The results of the benchmarking.

        Raises:
            ValueError: If an algorithm is not available,
                or is not suited to a problem configuration when this can be told
                before the execution,
                or if an option specific to a problem
                is not an option of the algorithm or has an invalid value.
                This is checked for all the combinations
                of algorithm configuration and problem configuration
                before any of them is executed.
            BenchmarkingError: If `raise_errors` is `True`
                and some executions raised exceptions.
                The other executions are not interrupted,
                and their performance histories are saved before the error is raised.
        """
        problem_configurations = tuple(problem_configurations)
        executions = self.__plan_executions(
            problem_configurations,
            algorithm_configurations,
            overwrite_histories,
        )
        if overwrite_histories:
            for algorithm_configuration in algorithm_configurations:
                for problem_configuration in problem_configurations:
                    self._results.remove_paths(
                        algorithm_configuration.name, problem_configuration.name
                    )

        if save_log and use_threading:
            # Set one file handler for all threads.
            file_handler = logging.FileHandler(
                self.__histories_path / "gemseo.log", "w"
            )
            file_handler.setFormatter(logging.Formatter("%(threadName)s %(message)s"))
            loggers = [LOGGER, gemseo_logger]
            for logger in loggers:
                logger.addHandler(file_handler)

        try:
            future_to_path = {}
            failures: dict[str, BaseException] = {}
            executor_class = (
                ThreadPoolExecutor if use_threading else ProcessPoolExecutor
            )
            with executor_class(max_workers=n_processes) as executor:
                for execution in executions:
                    future, path = self.__submit(
                        executor, execution, save_log, use_threading
                    )
                    future_to_path[future] = path

            # All the futures are done, as the executor has been shut down.
            # They are read in the order of submission, for a reproducible error.
            for future in future_to_path:
                exception = future.exception()
                if exception is None:
                    self._results.add_path(*future_to_path[future][1:])
                else:
                    description = future_to_path[future][0][:-1]
                    failures[description] = exception
                    LOGGER.error(
                        "%s raised: %s", description, exception, exc_info=exception
                    )
        finally:
            if save_log and use_threading:
                for logger in loggers:
                    logger.removeHandler(file_handler)

                file_handler.close()

        if future_to_path and self.__results_path:
            self._results.to_file(self.__results_path, 4)

        if failures:
            error = BenchmarkingError(failures, len(future_to_path))
            if raise_errors:
                raise error from next(iter(failures.values()))

            LOGGER.error("%s", error)

        return self._results

    def __plan_executions(
        self,
        problem_configurations: tuple[BaseProblemConfiguration, ...],
        algorithm_configurations: AlgorithmsConfigurations,
        overwrite_histories: bool,
    ) -> list[_Execution]:
        """Return the executions to submit, in the order of submission.

        This is the only place that decides which problems are solved,
        and with which settings.
        The problems whose performance histories already exist are skipped,
        unless these histories are overwritten.
        The algorithms and their settings are checked before any execution,
        so that a mistake in the last algorithm configuration
        does not come after the execution of the other ones.

        Args:
            problem_configurations: The problem configurations.
            algorithm_configurations: The algorithms configurations.
            overwrite_histories: Whether to overwrite the existing performance
                histories.

        Returns:
            The executions to submit.

        Raises:
            ValueError: If an algorithm is not available,
                or is not suited to a problem configuration when this can be told
                before the execution,
                or if an option specific to a problem
                is not an option of the algorithm or has an invalid value.
        """
        planned = []
        for algorithm_configuration in algorithm_configurations:
            for problem_index, problem_configuration in enumerate(
                problem_configurations
            ):
                for index in range(len(problem_configuration.starting_points)):
                    execution_info = (
                        f"problem {index + 1} "
                        f"of problem configuration {problem_configuration.name} "
                        f"for algorithm configuration {algorithm_configuration.name}"
                    )
                    if overwrite_histories or not self.__is_solved(
                        algorithm_configuration, problem_configuration, index
                    ):
                        planned.append((
                            algorithm_configuration,
                            problem_index,
                            index,
                            f"Solving {execution_info}.",
                        ))
                    else:
                        gemseo_logger.info("Skipping %s.", execution_info)

        self.__check_algorithms(problem_configurations, planned)
        executions = []
        messages = []
        for algorithm_configuration, problem_index, index, message in planned:
            problem_configuration = problem_configurations[problem_index]
            try:
                settings = algorithm_configuration.get_instance_settings(
                    problem_configuration, index
                )
            except Exception as error:  # ruff: ignore[blind-except]
                # Any error is reported,
                # as a callable defining an option specific to the problems
                # or a validator of the settings can raise any exception.
                messages.append(
                    f"The options of algorithm configuration "
                    f"{algorithm_configuration.name} "
                    f"for problem {index + 1} "
                    f"of problem configuration {problem_configuration.name} "
                    f"are invalid: {error}"
                )
            else:
                executions.append(
                    _Execution(
                        algorithm_configuration.name,
                        problem_configuration,
                        index,
                        settings,
                        message,
                    )
                )

        if messages:
            raise ValueError("\n".join(messages))

        return executions

    @staticmethod
    def __check_algorithms(
        problem_configurations: tuple[BaseProblemConfiguration, ...],
        planned: Iterable[tuple[AlgorithmConfiguration, int, int, str]],
    ) -> None:
        """Check the algorithms for the problem configurations to be solved.

        All the mistakes are reported at once.

        Args:
            problem_configurations: The problem configurations.
            planned: The algorithm configuration,
                the 0-based index of the problem configuration,
                the 0-based index of the problem
                and the message to log
                of each execution to submit.

        Raises:
            ValueError: If an algorithm is not available,
                or is not suited to a problem configuration.
        """
        algorithm_names = {index: {} for index in range(len(problem_configurations))}
        for algorithm_configuration, problem_index, _, _ in planned:
            algorithm_name = algorithm_configuration.algorithm_name
            algorithm_names[problem_index][algorithm_name] = None

        messages = {}
        for problem_index, problem_configuration in enumerate(problem_configurations):
            worker = problem_configuration.worker
            available_names = []
            for algorithm_name in algorithm_names[problem_index]:
                message = Benchmarker.__get_error_message(
                    worker.check_algorithm_availability, algorithm_name
                )
                if message:
                    messages[message] = None
                else:
                    available_names.append(algorithm_name)

            if available_names:
                message = Benchmarker.__get_error_message(
                    worker.check_algorithms_suitability,
                    available_names,
                    problem_configuration,
                )
                if message:
                    messages[message] = None

        if messages:
            raise ValueError("\n".join(messages))

    @staticmethod
    def __get_error_message(check: Callable[..., None], *args: Any) -> str:
        """Return the message of the error raised by a check.

        Args:
            check: The function that checks something and raises a `ValueError`
                if the check fails.
            *args: The arguments of the function.

        Returns:
            The message of the error, or an empty string if the check passed.
        """
        try:
            check(*args)
        except ValueError as error:
            return str(error)

        return ""

    def __submit(
        self,
        executor: Executor,
        execution: _Execution,
        save_log: bool,
        use_threading: bool,
    ) -> tuple[Future, tuple[str, str, str, Path]]:
        """Submit an execution of an algorithm configuration on a problem.

        Args:
            executor: The executor for parallel execution.
            execution: The execution.
            save_log: Whether to save the log to a file
                next to the performance history file.
            use_threading: Whether threads are used instead of processes
                to parallelize the execution.

        Returns:
            The future of the execution,
            and the message logged when solving the problem,
            the name of the algorithm configuration,
            the name of the problem configuration
            and the path to the performance history.
        """
        algorithm_configuration = AlgorithmConfiguration(
            execution.settings, execution.algorithm_configuration_name, {}
        )
        problem_configuration = execution.problem_configuration
        problem_configuration_name = problem_configuration.name
        index = execution.index
        performance_history_path = self.get_history_path(
            algorithm_configuration, problem_configuration_name, index, True
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
                index,
                "h5",
                True,
            )
        else:
            hdf_file_path = None

        future = executor.submit(
            problem_configuration.worker.execute,
            algorithm_configuration,
            problem_configuration,
            problem_configuration.starting_points[index],
            execution.message,
            gemseo_log_path,
            performance_history_path,
            hdf_file_path,
            gemseo_logger,
        )
        return future, (
            execution.message,
            execution.algorithm_configuration_name,
            problem_configuration_name,
            performance_history_path,
        )

    def __is_solved(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: BaseProblemConfiguration,
        index: int,
    ) -> bool:
        """Check whether a problem has already been solved.

        A problem is solved if the results contain
        the performance history of the algorithm configuration on the problem.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem_configuration: The problem configuration.
            index: The 0-based index of the problem.

        Returns:
            Whether the performance history of the problem is in the results.
        """
        return self._results.contains(
            algorithm_configuration.name,
            problem_configuration.name,
            self.get_history_path(
                algorithm_configuration, problem_configuration.name, index
            ),
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
                If `None`, the extension is for a JSON file.
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
