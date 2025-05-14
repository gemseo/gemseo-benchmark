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
"""Benchmarking worker for optimization algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo import execute_algo
from gemseo.algos.opt.factory import OptimizationLibraryFactory

from gemseo_benchmark.benchmarker.base_worker import BaseWorker
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.typing import RealArray
    from gemseo.utils.timer import Timer

    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )
    from gemseo_benchmark.problems.optimization_problem_configuration import (
        OptimizationBenchmarkingProblem,
    )


class OptimizationWorker(BaseWorker):
    """A benchmarking worker for optimization."""

    _algorithm_factory: OptimizationLibraryFactory = OptimizationLibraryFactory()

    def _get_problem(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: OptimizationBenchmarkingProblem,
        starting_point: RealArray,
        hdf_file_path: Path | None,
    ) -> OptimizationProblem:
        problem = problem_configuration.create_problem()
        problem.design_space.set_current_value(starting_point)
        return problem

    def _execute(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: OptimizationBenchmarkingProblem,
        starting_point: RealArray,
        problem: OptimizationProblem,
    ) -> None:
        execute_algo(
            problem,
            "opt",
            algo_name=algorithm_configuration.algorithm_name,
            **algorithm_configuration.algorithm_options,
        )

    def _create_performance_history(
        self,
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: OptimizationBenchmarkingProblem,
        problem: OptimizationProblem,
        timer: Timer,
    ) -> PerformanceHistory:
        performance_history = PerformanceHistory.from_problem(
            problem, problem_configuration.name
        )
        performance_history.algorithm_configuration = algorithm_configuration
        performance_history.total_time = timer.elapsed_time
        return performance_history

    def _post_execute(
        self, problem: OptimizationProblem, hdf_file_path: Path | None
    ) -> None:
        if hdf_file_path is not None:
            problem.database.to_hdf(hdf_file_path)
