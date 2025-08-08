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
"""Grouping of problem configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

from gemseo_benchmark.data_profiles.data_profile import DataProfile
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from pathlib import Path

    from gemseo_benchmark import ConfigurationPlotOptions
    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )
    from gemseo_benchmark.results.results import Results


class ProblemsGroup:
    """A group of problem configurations.

    !!! note

        Problem configurations should be grouped based on common characteristics such as
        functions smoothness and constraint set geometry.
    """

    name: str
    """The name of the group of problem configurations."""

    def __init__(
        self,
        name: str,
        problems: Iterable[BaseProblemConfiguration],
        description: str = "",
    ) -> None:
        """
        Args:
            name: The name of the group of problem configurations.
            problems: The problem configurations of the group.
            description: The description of the group of problem configurations.
        """  # noqa: D205, D212, D415
        self.name = name
        self.__problems = problems
        self.description = description

    def __iter__(self) -> Iterator[BaseProblemConfiguration]:
        return iter(self.__problems)

    # FIXME: Not suited to MDO and MDA?
    def compute_target_values(
        self,
        targets_number: int,
        algorithm_configurations: AlgorithmsConfigurations,
        only_feasible: bool = True,
    ) -> None:
        """Compute target values based on algorithm configurations.

        Args:
            targets_number: The number of target values to generate.
            algorithm_configurations: The algorithm configurations.
            only_feasible: Whether to generate only feasible target values.
        """
        for problem in self.__problems:
            problem.compute_target_values(
                targets_number, algorithm_configurations, only_feasible
            )

    def compute_data_profile(
        self,
        algos_configurations: AlgorithmsConfigurations,
        histories_paths: Results,
        show: bool = True,
        plot_path: str | Path = "",
        infeasibility_tolerance: float = 0.0,
        max_eval_number: int = 0,
        plot_settings: Mapping[str, ConfigurationPlotOptions] = READ_ONLY_EMPTY_DICT,
        grid_settings: Mapping[str, str] = READ_ONLY_EMPTY_DICT,
        use_abscissa_log_scale: bool = False,
    ) -> None:
        """Generate the data profiles of given algorithms relative to the problems.

        Args:
            algos_configurations: The algorithms configurations.
            histories_paths: The paths to the reference histories for each algorithm.
            show: If ``True``, show the plot.
            plot_path: The path where to save the plot.
                If empty, the plot is not saved.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum evaluations number to be displayed.
                If 0, this value is inferred from the longest history.
            plot_settings: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.
            grid_settings: The keyword arguments of `matplotlib.pyplot.grid`.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
        """
        data_profile = DataProfile({
            problem.name: problem.minimization_target_values
            for problem in self.__problems
        })

        for configuration_name in algos_configurations.names:
            for problem in self.__problems:
                for history_path in histories_paths.get_paths(
                    configuration_name, problem.name
                ):
                    history = PerformanceHistory.from_file(history_path)
                    if max_eval_number:
                        history = history.shorten(max_eval_number)
                    history.apply_infeasibility_tolerance(infeasibility_tolerance)
                    data_profile.add_history(
                        problem.name,
                        configuration_name,
                        history.performance_measures,
                        history.infeasibility_measures,
                    )

        data_profile.plot(
            show=show,
            file_path=plot_path,
            plot_settings=plot_settings,
            grid_settings=grid_settings,
            use_abscissa_log_scale=use_abscissa_log_scale,
        )

    def __len__(self) -> int:
        return len(self.__problems)
