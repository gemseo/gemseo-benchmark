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
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Grouping of reference problems for benchmarking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_benchmark.data_profiles.data_profile import DataProfile
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from pathlib import Path

    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.problems.problem import Problem
    from gemseo_benchmark.results.results import Results


class ProblemsGroup:
    """A group of reference problems for benchmarking.

    .. note::

       Reference problems should be grouped based on common characteristics such as
       functions smoothness and constraint set geometry.

    Attributes:
        name (str): The name of the group of problems.
    """

    def __init__(
        self,
        name: str,
        problems: Iterable[Problem],
        description: str = "",
    ) -> None:
        """
        Args:
            name: The name of the group of problems.
            problems: The benchmarking problems of the group.
            description: The description of the group of problems.
        """  # noqa: D205, D212, D415
        self.name = name
        self.__problems = problems
        self.description = description

    def __iter__(self) -> Iterator[Problem]:
        return iter(self.__problems)

    def is_algorithm_suited(self, name: str) -> bool:
        """Check whether an algorithm is suited to all the problems in the group.

        Args:
            name: The name of the algorithm.

        Returns:
            True if the algorithm is suited.
        """
        return all(problem.is_algorithm_suited(name) for problem in self.__problems)

    def compute_targets(
        self,
        targets_number: int,
        ref_algos_configurations: AlgorithmsConfigurations,
        only_feasible: bool = True,
    ) -> None:
        """Generate targets for all the problems based on given reference algorithms.

        Args:
            targets_number: The number of targets to generate.
            ref_algos_configurations: The configurations of the reference algorithms.
            only_feasible: Whether to generate only feasible targets.
        """
        for problem in self.__problems:
            problem.compute_targets(
                targets_number, ref_algos_configurations, only_feasible
            )

    def compute_data_profile(
        self,
        algos_configurations: AlgorithmsConfigurations,
        histories_paths: Results,
        show: bool = True,
        plot_path: str | Path | None = None,
        infeasibility_tolerance: float = 0.0,
        max_eval_number: int | None = None,
    ) -> None:
        """Generate the data profiles of given algorithms relative to the problems.

        Args:
            algos_configurations: The algorithms configurations.
            histories_paths: The paths to the reference histories for each algorithm.
            show: If True, show the plot.
            plot_path: The path where to save the plot.
                By default the plot is not saved.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum evaluations number to be displayed.
                If ``None``, this value is inferred from the longest history.
        """
        # Initialize the data profile
        target_values = {
            problem.name: problem.target_values for problem in self.__problems
        }
        data_profile = DataProfile(target_values)

        # Generate the performance histories
        for configuration_name in algos_configurations.names:
            for problem in self.__problems:
                for history_path in histories_paths.get_paths(
                    configuration_name, problem.name
                ):
                    history = PerformanceHistory.from_file(history_path)
                    if max_eval_number is not None:
                        history = history.shorten(max_eval_number)
                    history.apply_infeasibility_tolerance(infeasibility_tolerance)
                    data_profile.add_history(
                        problem.name,
                        configuration_name,
                        history.objective_values,
                        history.infeasibility_measures,
                    )

        # Plot and/or save the data profile
        data_profile.plot(show=show, file_path=plot_path)
