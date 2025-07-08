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

"""Benchmarking worker for multidisciplinary optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.opt.factory import OptimizationLibraryFactory

from gemseo_benchmark.benchmarker.base_worker import BaseWorker
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.typing import RealArray
    from gemseo.utils.timer import Timer

    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )
    from gemseo_benchmark.problems.mdo_problem_configuration import (
        MDOProblemConfiguration,
    )
    from gemseo_benchmark.problems.mdo_problem_configuration import MDOProblemType


class MDOWorker(BaseWorker):
    """A benchmarking worker for multidisciplinary optimization."""

    _algorithm_factory: OptimizationLibraryFactory = OptimizationLibraryFactory()

    @staticmethod
    def _get_problem(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: MDOProblemConfiguration,
        starting_point: RealArray,
        hdf_file_path: Path | None,
    ) -> MDOProblemType:
        scenario, disciplines = problem_configuration.create_problem(
            algorithm_configuration
        )
        scenario.formulation.optimization_problem.design_space.set_current_value(
            starting_point
        )
        return scenario, disciplines

    @staticmethod
    def _execute(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: MDOProblemConfiguration,
        starting_point: RealArray,
        problem: MDOProblemType,
    ) -> None:
        problem[0].execute()

    @staticmethod
    def _create_performance_history(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: MDOProblemConfiguration,
        problem: MDOProblemType,
        timer: Timer,
    ) -> PerformanceHistory:
        # TODO: Store execution times and number of discipline executions.
        performance_history = PerformanceHistory.from_problem(
            problem[0].formulation.optimization_problem, problem_configuration.name
        )
        performance_history.algorithm_configuration = algorithm_configuration
        performance_history.total_time = timer.elapsed_time
        return performance_history

    @staticmethod
    def _post_execute(problem: MDOProblemType, hdf_file_path: Path | None) -> None:
        if hdf_file_path is not None:
            problem[0].formulation.optimization_problem.database.to_hdf(hdf_file_path)
