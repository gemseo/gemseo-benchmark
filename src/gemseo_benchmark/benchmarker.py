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
import sys
from pathlib import Path
from typing import Iterable
from typing import Tuple

from gemseo.algos.database import Database
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import configure_logger
from gemseo.api import execute_algo
from gemseo.utils.timer import Timer
from gemseo_benchmark import join_substrings
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import \
    AlgorithmsConfigurations
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.results.performance_history import PerformanceHistory
from gemseo_benchmark.results.results import Results

LOGGER = configure_logger()


class Benchmarker(object):
    """A benchmarker of optimization algorithms on reference problems."""
    _HISTORY_CLASS = PerformanceHistory

    def __init__(
        self,
        histories_path: Path,
        results_path: Path = None,
        databases_path: Path = None,
        pseven_logs_path: Path = None
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
            pseven_logs_path: The path to the destination directory for the pSeven
                log files.
                If ``None``, the pSeven log files will not be saved.
        """
        self._databases_path = databases_path
        self.__histories_path = histories_path
        self.__is_algorithm_available = OptimizersFactory().is_available
        self.__pseven_logs_path = pseven_logs_path
        self.__results_path = results_path
        if results_path is not None and results_path.is_file():
            self._results = Results(results_path)
        else:
            self._results = Results()

    def execute(
        self,
        problems: Iterable[Problem],
        algorithms: AlgorithmsConfigurations,
        overwrite_histories: bool = False
    ) -> Results:
        """Run optimization algorithms on reference problems.

        Args:
            problems: The benchmarking problems.
            algorithms: The algorithms configurations.
            overwrite_histories: Whether to overwrite the existing performance
                histories.

        Returns:
            The results of the optimization.

        Raises:
            ValueError: If the algorithm is not available.
        """
        for algorithm_configuration in algorithms:

            algorithm_name = algorithm_configuration.algorithm_name
            if not self.__is_algorithm_available(algorithm_name):
                raise ValueError(f"The algorithm is not available: {algorithm_name}.")

            # Run the algorithm
            algorithm_configuration = self.__disable_stopping_criteria(
                algorithm_configuration
            )
            for problem in problems:
                self._solve_problem(
                    problem,
                    algorithm_configuration,
                    overwrite_histories=overwrite_histories
                )

        return self._results

    @staticmethod
    def __disable_stopping_criteria(
        algorithm_configuration: AlgorithmConfiguration
    ) -> AlgorithmConfiguration:
        """Disable the stopping criteria.

        Args:
            algorithm_configuration: The algorithm configuration.

        Returns:
            A copy of the algorithm configuration with disabled stopping criteria.
        """
        options = {
            "xtol_rel": 0.0,
            "xtol_abs": 0.0,
            "ftol_rel": 0.0,
            "ftol_abs": 0.0,
            "stop_crit_n_x": sys.maxsize,
        }
        options.update(algorithm_configuration.algorithm_options)
        return AlgorithmConfiguration(
            algorithm_configuration.algorithm_name,
            algorithm_configuration.name,
            **options
        )

    def _solve_problem(
        self,
        problem: Problem,
        algorithm_configuration: AlgorithmConfiguration,
        overwrite_histories: bool
    ) -> None:
        """Solve a benchmarking problem for all its starting points.

        Args:
            problem: The benchmarking problem.
            algorithm_configuration: The algorithm configuration.
            overwrite_histories: Whether to overwrite existing performance histories.
        """
        # Run an optimization from each starting point
        for problem_instance_index, problem_instance in enumerate(problem):

            if self.__skip_instance(
                    algorithm_configuration,
                    problem,
                    problem_instance_index,
                    overwrite_histories
            ):
                continue

            algorithm_configuration_copy = self.__set_pseven_log_file(
                algorithm_configuration, problem, problem_instance_index
            )
            database, history = self._run_algorithm(
                problem_instance,
                algorithm_configuration_copy,
                problem.name,
                problem_instance_index
            )

            self._save_history(history, problem_instance_index)

            if self._databases_path is not None:
                self.__save_database(
                    database,
                    algorithm_configuration,
                    problem.name,
                    problem_instance_index
                )

            if self.__results_path:
                self._results.to_file(self.__results_path, indent=4)

    def __skip_instance(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        bench_problem: Problem,
        index: int,
        overwrite_histories: bool
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
                self.__get_history_path(algorithm_configuration, problem_name, index)
        ):
            LOGGER.info(
                f"Skipping instance {instance} of problem {problem_name} for algorithm "
                f"configuration {algorithm_configuration.name}."
            )
            return True

        LOGGER.info(
            f"Solving instance {instance} of problem {problem_name} with algorithm "
            f"configuration {algorithm_configuration.name}."
        )
        return False

    def __set_pseven_log_file(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        problem: Problem,
        index: int
    ) -> AlgorithmConfiguration:
        """Copy an algorithm configuration by adding the path to the pSeven log file.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem: The benchmarking problem.
            index: The index of the problem instance.

        Returns:
            A copy of the configuration including the path to the pSeven log file.
        """
        if not self.__pseven_logs_path or not self.__is_algorithm_available("PSEVEN"):
            return algorithm_configuration

        from gemseo.algos.opt.lib_pseven import PSevenOpt
        if algorithm_configuration.algorithm_name not in PSevenOpt().descriptions:
            return algorithm_configuration

        return AlgorithmConfiguration(
            algorithm_configuration.algorithm_name,
            algorithm_configuration.name,
            **algorithm_configuration.algorithm_options,
            log_path=str(
                self.__get_pseven_log_path(
                    algorithm_configuration, problem.name, index
                )
            )
        )

    def _run_algorithm(
        self,
        problem: OptimizationProblem,
        algorithm_configuration: AlgorithmConfiguration,
        problem_name: str,
        index: int
    ) -> Tuple[Database, PerformanceHistory]:
        """Run an algorithm on a benchmarking problem for a particular starting point.

        Args:
            problem: The instance of the benchmarking problem.
            algorithm_configuration: The algorithm configuration.
            problem_name: The name of the benchmarking problem.
            index: The index of the problem instance.

        Returns:
            The runtime of the solver.
        """
        algo_name = algorithm_configuration.algorithm_name
        algo_options = algorithm_configuration.algorithm_options

        with Timer() as timer:
            execute_algo(problem, algo_name, **algo_options)

        history = self._HISTORY_CLASS.from_problem(problem, problem_name)
        history.algorithm_configuration = algorithm_configuration

        history.doe_size = 1
        if self.__is_algorithm_available("PSEVEN"):
            from gemseo.algos.opt.lib_pseven import PSevenOpt
            if algo_name in PSevenOpt().descriptions:
                history.doe_size = len(algo_options.get("sample_x", [None]))

        history.total_time = timer.elapsed_time
        return problem.database, history

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
        make_parents: bool = False
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
            make_parents=make_parents
        )

    def __get_pseven_log_path(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        problem_name: str,
        index: int
    ) -> Path:
        """Return a path for a pSeven log file.

        Args:
            algorithm_configuration: The algorithm configuration.
            problem_name: The name of the problem.
            index: The index of the problem instance.

        Returns:
            The path for the pSeven log file.

        Raises:
            ValueError: If the path to the destination directory for the
                pSeven files is not set.
        """
        if not self.__pseven_logs_path:
            raise ValueError("The directory for the pSeven files is not set.")

        return self._get_path(
            self.__pseven_logs_path,
            algorithm_configuration,
            problem_name,
            index,
            "txt",
            make_parents=True
        )

    @staticmethod
    def _get_path(
        root_dir: Path,
        algorithm_configuration: AlgorithmConfiguration,
        problem_name: str,
        index: int,
        extension: str = "json",
        make_parents: bool = False
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
        path = root_dir.resolve() / configuration_name / join_substrings(
            problem_name
        ) / f"{configuration_name}.{index + 1}.{extension}"
        if make_parents:
            path.parent.mkdir(parents=True, exist_ok=True)

        return path

    def __save_database(
        self,
        database: Database,
        algorithm_configuration: AlgorithmConfiguration,
        problem_name: str,
        index: int
    ) -> None:
        """Save the database of a problem.

        Args:
            database: The database.
            algorithm_configuration: The algorithm configuration.
            problem_name: The name of the problem.
            index: The index of the problem instance.
        """
        database.export_hdf(
                self._get_path(
                    self._databases_path,
                    algorithm_configuration,
                    problem_name,
                    index,
                    "h5",
                    make_parents=True
                )
        )
