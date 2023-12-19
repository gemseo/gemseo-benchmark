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
"""Class to compute data profiles for algorithms comparison.

A data profile is a graphical tool to compare iterative algorithms, e.g. optimization
algorithms or root-finding algorithms, on reference problems.

Each of the reference problems must be assigned targets, i.e. values of the objective
function or values of the residual norm, ranging from a first acceptable value to the
best known value for the problem.

The algorithms will be compared based on the number of targets they reach, cumulated
over all the reference problems, relative to the number of problems functions
evaluations they make.

The data profile is the empirical cumulated distribution function of the number of
functions evaluations made by an algorithm to reach a problem target.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
from gemseo.utils.matplotlib_figure import save_show_figure
from numpy import array
from numpy import linspace
from numpy import ndarray
from numpy import zeros

from gemseo_benchmark import COLORS_CYCLE
from gemseo_benchmark import MarkeveryType
from gemseo_benchmark import get_markers_cycle
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from numbers import Number
    from pathlib import Path

    from matplotlib.figure import Figure

    from gemseo_benchmark.data_profiles.target_values import TargetValues


class DataProfile:
    """Data profile that compares iterative algorithms on reference problems.

    A data profile is an empirical cumulative distribution function of the number of
    problems functions evaluations required by an algorithm to reach a reference problem
    target.
    """

    def __init__(self, target_values: Mapping[str, TargetValues]) -> None:
        """
        Args:
            target_values: The target values of each of the reference problems.
        """  # noqa: D205, D212, D415
        self.__targets_number = 0
        self.target_values = target_values
        self.__values_histories = {}

    @property
    def target_values(self) -> dict[str, TargetValues]:
        """The target values of each reference problem.

        Target values are a scale of objective function values,
        ranging from an easily achievable one to the best known value.
        A data profile is computed by counting the number of targets reached by an
        algorithm at each iteration.

        Raises:
            TypeError: if the target values are not passed as a dictionary.
            ValueError: If the reference problems have different numbers of target
                values.
        """
        return self.__target_values

    @target_values.setter
    def target_values(self, target_values: Mapping[str, TargetValues]) -> None:
        if not isinstance(target_values, Mapping):
            raise TypeError("The target values be must passed as a mapping")

        targets_numbers = {len(pb_targets) for pb_targets in target_values.values()}
        if len(targets_numbers) != 1:
            raise ValueError(
                "The reference problems must have the same number of target values."
            )

        self.__target_values = dict(target_values)
        self.__targets_number = targets_numbers.pop()

    def add_history(
        self,
        problem_name: str,
        algorithm_configuration_name: str,
        objective_values: Sequence[float],
        infeasibility_measures: Sequence[float] | None = None,
        feasibility_statuses: Sequence[bool] | None = None,
    ) -> None:
        """Add a history of performance values.

        Args:
            problem_name: The name of the problem.
            algorithm_configuration_name: The name of the algorithm configuration.
            objective_values: A history of objective values.
                N.B. the value at index ``i`` is assumed to have been obtained with
                ``i+1`` evaluations.
            infeasibility_measures: A history of infeasibility measures.
                If ``None`` then measures are set to zero in case of feasibility and set
                to infinity otherwise.
            feasibility_statuses: A history of (boolean) feasibility statuses.
                If ``None`` then feasibility is always assumed.

        Raises:
            ValueError: If the problem name is not the name of a reference problem.
        """
        if problem_name not in self.__target_values:
            raise ValueError(f"{problem_name!r} is not the name of a reference problem")
        if algorithm_configuration_name not in self.__values_histories:
            self.__values_histories[algorithm_configuration_name] = {
                pb_name: [] for pb_name in self.__target_values
            }
        history = PerformanceHistory(
            objective_values, infeasibility_measures, feasibility_statuses
        )
        self.__values_histories[algorithm_configuration_name][problem_name].append(
            history
        )

    def plot(
        self,
        algo_names: Iterable[str] | None = None,
        show: bool = True,
        file_path: str | Path | None = None,
        markevery: MarkeveryType = 0.1,
    ) -> None:
        """Plot the data profiles of the required algorithms.

        Args:
            algo_names: The names of the algorithms.
                If ``None`` then all the algorithms are considered.
            show: If True, show the plot.
            file_path: The path where to save the plot.
                If ``None``, the plot is not saved.
            markevery: The sampling parameter for the markers of the plot.
                Refer to the Matplotlib documentation.
        """
        if algo_names is None:
            algo_names = ()

        data_profiles = self.compute_data_profiles(*algo_names)
        figure = self._plot_data_profiles(data_profiles, markevery)
        save_show_figure(figure, show, file_path)

    def compute_data_profiles(self, *algo_names: str) -> dict[str, list[Number]]:
        """Compute the data profiles of the required algorithms.

        For each algorithm, compute the cumulative distribution function of the number
        of evaluations required by the algorithm to reach a reference target.

        Args:
            algo_names: The names of the algorithms.
                If ``None`` then all the algorithms are considered.

        Returns:
            The data profiles.
        """
        data_profiles = {}
        if not algo_names:
            algo_names = self.__values_histories.keys()

        for name in algo_names:
            total_hits_history = self.__compute_hits_history(name)
            problems_number = len(self.__target_values)
            repeat_number = self.__get_repeat_number(name)
            targets_total = self.__targets_number * problems_number * repeat_number
            ratios = total_hits_history / targets_total
            data_profiles[name] = ratios.tolist()
        return data_profiles

    def __compute_hits_history(self, algo_name: str) -> ndarray:
        """Compute the history of the number of target hits of an algorithm.

        Args:
            algo_name: The name of the algorithm.

        Returns:
            The history of the number of target hits.
        """
        algo_histories = self.__values_histories[algo_name]

        # Compute the maximal size of an optimization history
        max_history_size = max([
            max([len(pb_history) for pb_history in algo_history])
            for algo_history in algo_histories.values()
        ])

        # Compute the history of the number of target hits across all optimizations
        total_hits_history = zeros(max_history_size)
        for pb_name, targets in self.__target_values.items():
            for pb_history in algo_histories[pb_name]:
                hits_history = targets.compute_target_hits_history(pb_history)
                # If the history is shorter than the longest one, repeat its last value
                if len(hits_history) < max_history_size:
                    tail = [hits_history[-1]] * (max_history_size - len(hits_history))
                    hits_history.extend(tail)

                total_hits_history += array(hits_history)

        return total_hits_history

    def __get_repeat_number(self, algo_name: str) -> int:
        """Check that an algorithm has the same number of histories for each problem.

        Make sure that the reference problems are equally represented with respect to
        the algorithm performance.

        Args:
            algo_name: The name of the algorithm.

        Returns:
            The common number of values histories per problem.

        Raises:
            ValueError: If the algorithm does not have the same number of histories
                for each problem.
        """
        histories_numbers = {
            len(histories) for histories in self.__values_histories[algo_name].values()
        }
        if len(histories_numbers) != 1:
            raise ValueError(
                f"Reference problems unequally represented for algorithm {algo_name!r}."
            )
        return histories_numbers.pop()

    @staticmethod
    def _plot_data_profiles(
        data_profiles: Mapping[str, Sequence[Number]], markevery: MarkeveryType = 0.1
    ) -> Figure:
        """Plot the data profiles.

        Args:
            data_profiles: The data profiles.
            markevery: The sampling parameter for the markers of the plot.
                Refer to the Matplotlib documentation.

        Returns:
            The data profiles figure.
        """
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)

        # Set the title and axes
        axes.set_title(f"Data profile{'s' if len(data_profiles) > 1 else ''}")
        max_profile_size = max([len(profile) for profile in data_profiles.values()])
        axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        plt.xlabel("Number of functions evaluations")
        plt.xlim([1, max_profile_size])
        y_ticks = linspace(0.0, 1.0, 11)
        plt.yticks(y_ticks, (f"{ratio * 100.0:02.0f}%" for ratio in y_ticks))
        plt.ylabel("Ratios of targets reached")
        plt.ylim([0.0, 1.05])

        # Plot the 100% line
        axes.axhline(1.0, linestyle=":", color="black")

        # Plot the data profiles
        for color, marker, (name, profile) in zip(
            COLORS_CYCLE, get_markers_cycle(), data_profiles.items()
        ):
            # Plot the data profile
            profile_size = len(profile)
            axes.plot(
                range(1, profile_size + 1),
                profile,
                color=color,
                label=name,
                marker=marker,
                markevery=markevery,
            )

            # Extend the profile with an horizontal line if necessary
            if profile_size < max_profile_size:
                tail_size = max_profile_size - profile_size + 1
                last_value = profile[-1]
                axes.plot(
                    range(profile_size, profile_size + tail_size),
                    [last_value] * tail_size,
                    color=color,
                    linestyle="dotted",
                )
                # Mark the last entry of the data profile
                axes.plot(profile_size, last_value, marker="*", color=color)
        plt.legend()

        return fig
