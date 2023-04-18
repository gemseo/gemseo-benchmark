"""A class to implement a benchmarking worker."""
from __future__ import annotations

from typing import Tuple

from gemseo.algos.database import Database
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import execute_algo
from gemseo.utils.timer import Timer

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.results.performance_history import PerformanceHistory

WorkerOutputs = Tuple[Problem, int, Database, PerformanceHistory]


class Worker:
    """A benchmarking worker."""

    def __init__(
        self,
        factory: OptimizersFactory,
        history_class: type[PerformanceHistory] = PerformanceHistory,
    ) -> None:
        """
        Args:
            factory: The factory for optimizers.
            history_class: The class of performance history.
        """  # noqa: D205, D212, D415
        self.__factory = factory
        self.__history_class = history_class

    def __call__(
        self, args: tuple[AlgorithmConfiguration, Problem, OptimizationProblem, int]
    ) -> WorkerOutputs:
        """Run an algorithm on a benchmarking problem for a particular starting point.

        Args:
            args:
                The algorithm configuration,
                the benchmarking problem,
                the instance of the benchmarking problem,
                the index of the problem instance.

        Returns:
            The database of the algorithm run and its performance history.
        """
        (
            algorithm_configuration,
            problem,
            problem_instance,
            problem_instance_index,
        ) = args
        algo_name = algorithm_configuration.algorithm_name
        algo_options = algorithm_configuration.algorithm_options
        with Timer() as timer:
            execute_algo(problem_instance, algo_name, **algo_options)

        history = self.__history_class.from_problem(problem_instance, problem.name)
        history.algorithm_configuration = algorithm_configuration
        history.doe_size = 1
        if self.__factory.is_available("PSEVEN"):
            from gemseo.algos.opt.lib_pseven import PSevenOpt

            if algo_name in PSevenOpt().descriptions:
                history.doe_size = len(algo_options.get("sample_x", [None]))

        history.total_time = timer.elapsed_time
        return problem, problem_instance_index, problem_instance.database, history
