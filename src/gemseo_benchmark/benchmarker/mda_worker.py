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

"""Benchmarking worker for multidisciplinary analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.discipline.base_discipline import CacheType
from gemseo.mda.factory import MDAFactory

from gemseo_benchmark.benchmarker._metrics import DisciplineExecutions
from gemseo_benchmark.benchmarker._metrics import ElapsedTime
from gemseo_benchmark.benchmarker.base_worker import BaseWorker
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.typing import RealArray
    from gemseo.utils.timer import Timer

    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )
    from gemseo_benchmark.problems.mda_problem_configuration import (
        MDAProblemConfiguration,
    )
    from gemseo_benchmark.problems.mda_problem_configuration import MDAProblemType


class MDAWorker(BaseWorker):
    """A benchmarking worker for multidisciplinary analysis."""

    _algorithm_factory: MDAFactory = MDAFactory()

    @staticmethod
    def _get_problem(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: MDAProblemConfiguration,
        starting_point: RealArray,
        hdf_file_path: Path | None,
    ) -> MDAProblemType:
        mda, disciplines = problem_configuration.create_problem(algorithm_configuration)
        if hdf_file_path is not None:
            mda.set_cache(CacheType.HDF5, hdf_file_path=hdf_file_path)

        return mda, disciplines

    @classmethod
    def _add_metrics_listeners(
        cls, problem: MDAProblemType
    ) -> tuple[ElapsedTime, DisciplineExecutions]:
        mda_solver = problem[0]
        elapsed_time = ElapsedTime()
        mda_solver.add_iteration_callback(elapsed_time.add_metrics)
        discipline_executions = DisciplineExecutions(problem[1])
        mda_solver.add_iteration_callback(discipline_executions.add_metrics)
        return elapsed_time, discipline_executions

    @staticmethod
    def _execute(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: MDAProblemConfiguration,
        starting_point: RealArray,
        problem: MDAProblemType,
    ) -> None:
        problem[0].execute(
            problem_configuration.variable_space.convert_array_to_dict(starting_point)
        )

    @staticmethod
    def _create_performance_history(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: MDAProblemConfiguration,
        problem: MDAProblemType,
        timer: Timer,
        metrics_listeners: tuple[ElapsedTime, DisciplineExecutions],
    ) -> PerformanceHistory:
        time_listener, discipline_listener = metrics_listeners
        return PerformanceHistory(
            problem[0].residual_history,
            problem_name=problem_configuration.name,
            total_time=timer.elapsed_time,
            algorithm_configuration=algorithm_configuration,
            number_of_variables=problem_configuration.dimension,
            elapsed_times=time_listener.get_metrics(timer),
            number_of_discipline_executions=discipline_listener.get_metrics(),
        )

    @staticmethod
    def _post_execute(problem: MDAProblemType, hdf_file_path: Path | None) -> None:
        pass
