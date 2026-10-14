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

import logging
from typing import TYPE_CHECKING

from gemseo import execute_algo
from gemseo.optimization.factory import OptimizationLibraryFactory

from gemseo_benchmark.benchmarker._metrics import ElapsedTime
from gemseo_benchmark.benchmarker.base_worker import BaseWorker
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from gemseo.optimization import OptimizationProblem
    from gemseo.util.timer import Timer
    from gemseo.util.typing import RealArray

    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )
    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )
    from gemseo_benchmark.problems.optimization_problem_configuration import (
        OptimizationProblemConfiguration,
    )

LOGGER = logging.getLogger(__name__)


class OptimizationWorker(
    BaseWorker[
        "OptimizationProblemConfiguration", "OptimizationProblem", tuple[ElapsedTime]
    ]
):
    """A benchmarking worker for optimization."""

    _algorithm_factory: OptimizationLibraryFactory = OptimizationLibraryFactory()

    @classmethod
    def check_algorithms_suitability(
        cls,
        algorithm_names: Iterable[str],
        problem_configuration: BaseProblemConfiguration,
    ) -> None:
        """Check whether algorithms are suited to an optimization problem.

        An algorithm is suited if GEMSEO says that it is for a problem created by
        the problem configuration, e.g. an algorithm that does not handle constraints
        is not suited to a problem that has some.

        If the problem cannot be created, the check is skipped:
        the exception is raised and reported by the execution that needs the problem.
        """
        try:
            problem = problem_configuration.create_problem()
        except Exception as error:  # noqa: BLE001
            LOGGER.warning(
                "The suitability of the algorithms to the problem configuration %r "
                "is not checked, as the problem cannot be created: %s",
                problem_configuration.name,
                error,
            )
            return

        factory = cls._algorithm_factory
        reasons = {}
        for algorithm_name in algorithm_names:
            library_class = factory.get_class(
                factory.algo_name_to_library[algorithm_name]
            )
            description = library_class.ALGORITHM_INFOS[algorithm_name]
            # There is no public accessor to the reason in GEMSEO.
            reason = library_class._get_unsuitability_reason(  # noqa: SLF001
                description, problem
            )
            if reason:
                reasons[repr(algorithm_name)] = reason

        if reasons:
            if len(reasons) == 1:
                ((algorithm_name, reason),) = reasons.items()
                subject = f"The algorithm {algorithm_name} is"
                details = f" {reason}"
            else:
                subject = f"The algorithms {', '.join(reasons)} are"
                details = ",".join(
                    f"\n- {name}: {reason}" for name, reason in reasons.items()
                )

            msg = (
                f"{subject} not adapted "
                f"to the problem configuration {problem_configuration.name!r}:"
                f"{details}."
            )
            raise ValueError(msg)

    @staticmethod
    def _get_problem(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: OptimizationProblemConfiguration,
        starting_point: RealArray,
        hdf_file_path: Path | None,
    ) -> OptimizationProblem:
        problem = problem_configuration.create_problem()
        # TODO(bump-gemseo): cannot transform: the type of problem.design_space could not be inferred; if it is an instance of ParameterSpace, a current value is a DesignSpace notion; use nominal_value to read the reference point of a space, which is the mean of the distributions here  # noqa: E501
        problem.design_space.set_current_value(starting_point)
        return problem

    @classmethod
    def _add_metrics_listeners(cls, problem: OptimizationProblem) -> tuple[ElapsedTime]:
        elapsed_time = ElapsedTime()
        problem.add_listener(elapsed_time.add_metrics)
        return (elapsed_time,)

    @staticmethod
    def _execute(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: OptimizationProblemConfiguration,
        starting_point: RealArray,
        problem: OptimizationProblem,
    ) -> None:
        execute_algo(
            problem,
            "opt",
            settings_model=algorithm_configuration.algorithm_settings,
        )

    @staticmethod
    def _create_performance_history(
        algorithm_configuration: AlgorithmConfiguration,
        problem_configuration: OptimizationProblemConfiguration,
        problem: OptimizationProblem,
        timer: Timer,
        metrics_listeners: tuple[ElapsedTime],
    ) -> PerformanceHistory:
        (time_listener,) = metrics_listeners
        performance_history = PerformanceHistory.from_problem(
            problem,
            problem_configuration.name,
            time_listener.get_metrics(timer),
        )
        performance_history.algorithm_configuration = algorithm_configuration
        performance_history.total_time = timer.elapsed_time
        return performance_history

    @staticmethod
    def _post_execute(problem: OptimizationProblem, hdf_file_path: Path | None) -> None:
        if hdf_file_path is not None:
            problem.database.to_hdf(hdf_file_path)
