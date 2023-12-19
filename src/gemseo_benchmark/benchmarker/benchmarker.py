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
"""A benchmarker of optimization algorithms on reference problems."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import Final

from gemseo import configure_logger
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)

from gemseo_benchmark import join_substrings
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.benchmarker.worker import Worker
from gemseo_benchmark.benchmarker.worker import WorkerOutputs
from gemseo_benchmark.results.performance_history import PerformanceHistory
from gemseo_benchmark.results.results import Results

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from gemseo.algos.database import Database

    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.problems.problem import Problem

LOGGER = configure_logger()


class Benchmarker:
    """A benchmarker of optimization algorithms on reference problems."""

    _HISTORY_CLASS: Final = PerformanceHistory

    def __init__(
        self,
        histories_path: Path,
        results_path: Path | None = None,
        databases_path: Path | None = None,
    ) -> None:
        """
        Args:
            histories_path: The path to the directory where to save the performance
                histories.
            results_path: The path to the file for saving the performance histories
                paths.
                If exists, the file is updated with the new performance histories paths.
            databases_path: The path to the destination directory for the databases.
                If ``None``, the databases will not be saved.
        """  # noqa: D205, D212, D415
        self._databases_path = databases_path
        self.__histories_path = histories_path
        self.__optimizers_factory = OptimizersFactory()
        self.__is_algorithm_available = self.__optimizers_factory.is_available
        self.__results_path = results_path
        if results_path is not None and results_path.is_file():
            self._results = Results(results_path)
        else:
            self._results = Results()

    def execute(
        self,
        problems: Iterable[Problem],
        algorithms: AlgorithmsConfigurations,
        overwrite_histories: bool = False,
        number_of_processes: int = 1,
        use_threading: bool = False,
    ) -> Results:
        """Run optimization algorithms on reference problems.

        Args:
            problems: The benchmarking problems.
            algorithms: The algorithms configurations.
            overwrite_histories: Whether to overwrite the existing performance
                histories.
            number_of_processes: The maximum simultaneous number of threads or
                processes used to parallelize the execution.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution.

        Returns:
            The results of the optimization.

        Raises:
            ValueError: If the algorithm is not available.
        """
        # Prepare the inputs of the benchmarking workers
        inputs = []
        for algorithm_configuration in [config.copy() for config in algorithms]:
            algorithm_name = algorithm_configuration.algorithm_name
            if not self.__is_algorithm_available(algorithm_name):
                raise ValueError(f"The algorithm is not available: {algorithm_name}.")

            self.__disable_stopping_criteria(algorithm_configuration)
            for problem in problems:
                inputs.extend([
                    (
                        self.__set_instance_algorithm_options(
                            algorithm_configuration,
                            problem_instance_index,
                        ),
                        problem,
                        problem_instance,
                        problem_instance_index,
                    )
                    for problem_instance_index, problem_instance in enumerate(problem)
                    if not self.__skip_instance(
                        algorithm_configuration,
                        problem,
                        problem_instance_index,
                        overwrite_histories,
                    )
                ])

        if inputs:
            worker = Worker(self._HISTORY_CLASS)
            if number_of_processes == 1:
                for worker_inputs in inputs:
                    self.__worker_callback(0, worker(worker_inputs))
            else:
                CallableParallelExecution(
                    [worker],
                    number_of_processes,
                    use_threading,
                ).execute(inputs, self.__worker_callback)

        return self._results

    @staticmethod
    def __disable_stopping_criteria(
        algorithm_configuration: AlgorithmConfiguration,
    ) -> None:
        """Disable the stopping criteria.

        Args:
            algorithm_configuration: The algorithm configuration.
        """
        algorithm_configuration.algorithm_options.update({
            "xtol_rel": 0.0,
            "xtol_abs": 0.0,
            "ftol_rel": 0.0,
            "ftol_abs": 0.0,
            "stop_crit_n_x": sys.maxsize,
        })

    def __skip_instance(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        bench_problem: Problem,
        index: int,
        overwrite_histories: bool,
    ) -> bool:
        """Check whether a problem instance has already been solved.

        Args:
            algorithm_configuration: The algorithm configuration.
            bench_problem: The benchmarking problem.
            index: The index of the instance.
            overwrite_histories: Whether to overwrite existing histories.

        Returns:
            Whether to solve the problem instance.
        """
        instance = index + 1
        problem_name = bench_problem.name

        if not overwrite_histories and self._results.contains(
            algorithm_configuration.name,
            problem_name,
            self.__get_history_path(algorithm_configuration, problem_name, index),
        ):
            LOGGER.info(
                "Skipping instance %s of problem %s for algorithm configuration %s.",
                instance,
                problem_name,
                algorithm_configuration.name,
            )
            return True

        LOGGER.info(
            "Solving instance %s of problem %s with algorithm configuration %s.",
            instance,
            problem_name,
            algorithm_configuration.name,
        )
        return False

    @staticmethod
    def __set_instance_algorithm_options(
        algorithm_configuration: AlgorithmConfiguration,
        index: int,
    ) -> AlgorithmConfiguration:
        """Return the algorithm configuration of an instance of a problem.

        Args:
            algorithm_configuration: The algorithm configuration.
            index: The 0-based index of the problem instance.

        Returns:
            The algorithm configuration of the problem instance.
        """
        algorithm_options = dict(algorithm_configuration.algorithm_options)
        for name, value in algorithm_configuration.instance_algorithm_options.items():
            algorithm_options[name] = value(index)

        return AlgorithmConfiguration(
            algorithm_configuration.algorithm_name,
            algorithm_configuration.name,
            {},
            **algorithm_options,
        )

    def __worker_callback(self, _: int, outputs: WorkerOutputs) -> None:
        """Save the history and database of a benchmarking worker.

        Args:
            _: The index of the worker.
            outputs: The outputs of the worker.
        """
        problem, problem_instance_index, database, history = outputs
        self._save_history(history, problem_instance_index)
        if self._databases_path is not None:
            self.__save_database(
                database,
                history.algorithm_configuration,
                problem.name,
                problem_instance_index,
            )

        if self.__results_path:
            self._results.to_file(self.__results_path, indent=4)

    def _save_history(self, history: PerformanceHistory, index: int) -> None:
        """Save a performance history into a history file.

        Args:
            history: The performance history.
            index: The index of the problem instance.
        """
        problem_name = history.problem_name
        algorithm_configuration = history.algorithm_configuration
        path = self.__get_history_path(
            algorithm_configuration, problem_name, index, make_parents=True
        )
        history.to_file(path)
        self._results.add_path(algorithm_configuration.name, problem_name, path)

    def __get_history_path(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        problem_name: str,
        index: int,
        make_parents: bool = False,
    ) -> Path:
        """Return a path for a history file.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem_name: The name of the problem.
            index: The index of the problem instance.
            make_parents: Whether to make the parent directories.

        Returns:
            The path for the history file.
        """
        return self._get_path(
            self.__histories_path,
            algorithm_configuration,
            problem_name,
            index,
            "json",
            make_parents=make_parents,
        )

    @staticmethod
    def _get_path(
        root_dir: Path,
        algorithm_configuration: AlgorithmConfiguration,
        problem_name: str,
        index: int,
        extension: str = "json",
        make_parents: bool = False,
    ) -> Path:
        """Return a path in the file tree dedicated to a specific optimization run.

        Args:
            root_dir: The path to the root directory.
            algorithm_configuration: The algorithm configuration.
            problem_name: The name of the problem.
            index: The index of the problem instance.
            extension: The extension of the path.
                If ``None``, the extension is for a JSON file.
            make_parents: Whether to make the parent directories of the path.

        Returns:
            The path for the file.
        """
        configuration_name = join_substrings(algorithm_configuration.name)
        path = (
            root_dir.resolve()
            / configuration_name
            / join_substrings(problem_name)
            / f"{configuration_name}.{index + 1}.{extension}"
        )
        if make_parents:
            path.parent.mkdir(parents=True, exist_ok=True)

        return path

    def __save_database(
        self,
        database: Database,
        algorithm_configuration: AlgorithmConfiguration,
        problem_name: str,
        index: int,
    ) -> None:
        """Save the database of a problem.

        Args:
            database: The database.
            algorithm_configuration: The algorithm configuration.
            problem_name: The name of the problem.
            index: The index of the problem instance.
        """
        database.to_hdf(
            self._get_path(
                self._databases_path,
                algorithm_configuration,
                problem_name,
                index,
                "h5",
                make_parents=True,
            )
        )
