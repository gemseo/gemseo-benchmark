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
"""A class to implement a benchmarking worker."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo import execute_algo
from gemseo.algos.database import Database
from gemseo.utils.timer import Timer

from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from gemseo.algos.opt_problem import OptimizationProblem

    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )

WorkerOutputs = tuple[Problem, int, Database, PerformanceHistory]


class Worker:
    """A benchmarking worker."""

    def __init__(
        self, history_class: type[PerformanceHistory] = PerformanceHistory
    ) -> None:
        """
        Args:
            history_class: The class of performance history.
        """  # noqa: D205, D212, D415
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
        history.total_time = timer.elapsed_time
        return problem, problem_instance_index, problem_instance.database, history
