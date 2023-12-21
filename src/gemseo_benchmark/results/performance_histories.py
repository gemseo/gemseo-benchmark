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
"""A class to implement a collection of performance histories."""

from __future__ import annotations

import collections.abc
import statistics
from typing import TYPE_CHECKING
from typing import Callable

import numpy

from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from gemseo_benchmark import MarkeveryType
    from gemseo_benchmark.results.history_item import HistoryItem


class PerformanceHistories(collections.abc.MutableSequence):
    """A collection of performance histories."""

    __histories: list[PerformanceHistory]
    """The performance histories of the collection."""

    def __init__(self, *histories: PerformanceHistory) -> None:
        """
        Args:
            *histories: The performance histories.
        """  # noqa: D205, D212, D415
        self.__histories = list(histories)

    def __getitem__(self, index: int) -> PerformanceHistory:
        return self.__histories[index]

    def __setitem__(self, index: int, history: PerformanceHistory) -> None:
        self.__histories[index] = history

    def __delitem__(self, index: int) -> None:
        del self.__histories[index]

    def __len__(self) -> int:
        return len(self.__histories)

    def insert(self, index: int, history: PerformanceHistory) -> None:
        """Insert a performance history in the collection.

        Args:
            index: The index where to insert the performance history.
            history: The performance history.
        """
        self.__histories.insert(index, history)

    def __get_equal_size_histories(self) -> PerformanceHistories:
        """Return the histories extended to the maximum size."""
        return PerformanceHistories(*[
            history.extend(self.__maximum_size) for history in self
        ])

    @property
    def __maximum_size(self) -> int:
        """The maximum size of a history."""
        return max(len(history) for history in self)

    def compute_minimum(self) -> PerformanceHistory:
        """Return the itemwise minimum history of the collection.

        Returns:
            The itemwise minimum history of the collection.
        """
        return self.__compute_itemwise_statistic(min)

    def compute_maximum(self) -> PerformanceHistory:
        """Return the itemwise maximum history of the collection.

        Returns:
            The itemwise maximum history of the collection.
        """
        return self.__compute_itemwise_statistic(max)

    def compute_median(self, compute_low_median: bool = True) -> PerformanceHistory:
        """Return the itemwise median history of the collection.

        Args:
            compute_low_median: Whether to compute the low median
                (rather than the high median).

        Returns:
            The itemwise median history of the collection.
        """
        if compute_low_median:
            return self.__compute_itemwise_statistic(statistics.median_low)

        return self.__compute_itemwise_statistic(statistics.median_high)

    def __compute_itemwise_statistic(
        self,
        statistic_computer: Callable[[tuple[HistoryItem]], HistoryItem],
    ) -> PerformanceHistory:
        """Return the history of an itemwise statistic of the collection.

        The histories are extended to the same length before being split.

        Args:
            statistic_computer: The computer of the statistic.

        Returns:
            The history of the itemwise statistic.
        """
        history = PerformanceHistory()
        history.items = [
            statistic_computer(items)
            for items in zip(*[
                history.items for history in self.__get_equal_size_histories()
            ])
        ]
        return history

    def cumulate_minimum(self) -> PerformanceHistories:
        """Return the histories of the minimum."""
        return PerformanceHistories(*[
            history.compute_cumulated_minimum() for history in self
        ])

    def plot_algorithm_histories(
        self,
        axes: Axes,
        algorithm_name: str,
        max_feasible_objective: float,
        plot_all: bool,
        color: str,
        marker: str,
        alpha: float,
        markevery: MarkeveryType,
    ) -> float | None:
        """Plot the histories associated with an algorithm.

        Args:
            axes: The axes on which to plot the performance histories.
            algorithm_name: The name of the algorithm.
            max_feasible_objective: The ordinate for infeasible history items.
            plot_all: Whether to plot all the performance histories.
            color: The color of the plot.
            marker: The marker type of the plot.
            alpha: The opacity level for overlapping areas.
                Refer to the Matplotlib documentation.
            markevery: The sampling parameter for the markers of the plot.
                Refer to the Matplotlib documentation.

        Returns:
            The minimum feasible objective value of the median history
            or ``None`` if the median history has no feasible item.
        """
        # Plot all the performance histories
        if plot_all:
            for history in self:
                history.plot(axes, only_feasible=True, color=color, alpha=alpha)

        # Get the minimum history, starting from its first feasible item
        abscissas, minimum_items = self.compute_minimum().get_plot_data(feasible=True)
        minimum_ordinates = [item.objective_value for item in minimum_items]

        # Get the maximum history for the same abscissas as the minimum history
        maximum_items = self.compute_maximum().items
        # Replace the infeasible objective values with the maximum value
        # N.B. Axes.fill_between requires finite values, that is why the infeasible
        # objective values are replaced with a finite value rather than with infinity.
        maximum_ordinates = self.__get_penalized_objective_values(
            maximum_items, abscissas, max_feasible_objective
        )

        # Plot the area between the minimum and maximum histories.
        axes.fill_between(abscissas, minimum_ordinates, maximum_ordinates, alpha=alpha)
        axes.plot(abscissas, minimum_ordinates, color=color, alpha=alpha)
        # Replace the infeasible objective values with infinity
        maximum_ordinates = self.__get_penalized_objective_values(
            maximum_items, abscissas, numpy.inf
        )
        axes.plot(abscissas, maximum_ordinates, color=color, alpha=alpha)

        # Plot the median history
        median = self.compute_median()
        median.plot(
            axes,
            only_feasible=True,
            label=algorithm_name,
            color=color,
            marker=marker,
            markevery=markevery,
        )

        # Return the smallest objective value of the median
        _, history_items = median.get_plot_data(feasible=True)
        if history_items:
            return min(history_items).objective_value
        return None

    @staticmethod
    def __get_penalized_objective_values(
        history_items: Sequence[HistoryItem], indexes: Iterable[int], value: float
    ) -> list[float]:
        """Return the objectives of history items, replacing the infeasible ones.

        Args:
            history_items: The history items.
            indexes: The 1-based indexes of the history items.
            value: The replacement for infeasible objective values.

        Returns:
            The objective values.
        """
        return [
            history_items[index - 1].objective_value
            if history_items[index - 1].is_feasible
            else value
            for index in indexes
        ]
