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
from matplotlib.ticker import MaxNLocator

from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
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

    def get_equal_size_histories(self) -> PerformanceHistories:
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
                history.items for history in self.get_equal_size_histories()
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

    def plot_performance_measure_distribution(
        self, axes: Axes, max_feasible_objective: float | None = None
    ) -> None:
        """Plot the distribution of the performance measure.

        Args:
            axes: The axes of the plot.
            max_feasible_objective: The maximum feasible objective value.
        """
        if max_feasible_objective is None:
            max_feasible_objective = max(
                item.objective_value
                for history in self
                for item in history
                if item.is_feasible
            )

        self.__plot_distribution(
            numpy.array([
                [
                    item.objective_value if item.is_feasible else numpy.nan
                    for item in history
                ]
                for history in self.get_equal_size_histories()
            ]),
            axes,
            "Performance measure",
            max_feasible_objective,
        )

    def plot_infeasibility_measure_distribution(self, axes: Axes) -> None:
        """Plot the distribution of the infeasibility measure.

        Args:
            axes: The axes of the plot.
        """
        self.__plot_distribution(
            numpy.array([history.infeasibility_measures for history in self]),
            axes,
            "Infeasibility measure",
        )

    def plot_number_of_unsatisfied_constraints_distribution(self, axes: Axes) -> None:
        """Plot the distribution of the number of unsatisfied constraints.

        Args:
            axes: The axes of the plot.
        """
        self.__plot_distribution(
            numpy.array([history.n_unsatisfied_constraints for history in self]),
            axes,
            "Number of unsatisfied constraints",
        )

    @staticmethod
    def __plot_distribution(
        histories: numpy.ndarray,
        axes: Axes,
        y_label: str,
        infinity: float | None = None,
    ) -> None:
        """Plot the distribution of histories data.

        Args:
            histories: The histories data.
            axes: The axes of the plot.
            y_label: The label for the vertical axis.
            infinity: The substitute value for infinite ordinates.
        """
        PerformanceHistories.plot_centiles_range(
            histories,
            axes,
            (0, 100),
            {"color": "lightgray", "label": "0th-100th centiles"},
            infinity,
        )
        PerformanceHistories.plot_centiles_range(
            histories,
            axes,
            (25, 75),
            {"color": "gray", "label": "25th-75th centiles"},
            infinity,
        )
        PerformanceHistories.plot_median(
            histories, axes, {"color": "black", "label": "median"}
        )
        axes.plot(
            range(1, histories.shape[1] + 1),
            numpy.mean(histories, 0),
            color="orange",
            label="mean",
            linestyle=":",
        )
        # Reorder the legend
        axes.legend(
            *zip(*[
                list(zip(*axes.get_legend_handles_labels()))[index]
                for index in [3, 2, 1, 0]
            ])
        )
        axes.set_xlabel("Number of functions evaluations")
        axes.set_ylabel(y_label)

    @staticmethod
    def plot_centiles_range(
        histories: numpy.ndarray,
        axes: Axes,
        centile_range: tuple[float, float],
        fill_between_kwargs: Mapping[str, str],
        infinity: float | None,
    ) -> None:
        """Plot a range of centiles of histories data.

        Args:
            histories: The histories data.
            axes: The axes of the plot.
            centile_range: The range of centiles to be drawn.
            fill_between_kwargs: Keyword arguments
                for `matplotlib.axes.Axes.fill_between`.
            infinity: The substitute value for infinite ordinates.
        """
        method = "inverted_cdf"  # supports numpy.inf
        histories = numpy.nan_to_num(histories, nan=numpy.inf)
        lower_centile = numpy.percentile(
            histories, min(centile_range), 0, method=method
        )
        first_index = next(
            (i for i, value in enumerate(lower_centile) if numpy.isfinite(value)),
            len(lower_centile),
        )
        axes.plot(  # hack to get same limits/ticks
            range(1, first_index + 1),
            numpy.full(
                first_index,
                lower_centile[first_index]
                if first_index < len(lower_centile)
                else numpy.nan,
            ),
            alpha=0,
        )
        upper_centile = numpy.percentile(
            histories[:, first_index:], max(centile_range), 0, method=method
        )

        if infinity is not None:
            upper_centile = numpy.nan_to_num(upper_centile, posinf=infinity)

        axes.fill_between(
            range(first_index + 1, histories.shape[1] + 1),
            lower_centile[first_index:],
            upper_centile,
            **fill_between_kwargs,
        )
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))

    @staticmethod
    def plot_median(
        histories: numpy.ndarray,
        axes: Axes,
        plot_kwargs: Mapping[str, str | int | float],
    ) -> None:
        """Plot a range of centiles of histories data.

        Args:
            histories: The histories data.
            axes: The axes of the plot.
            plot_kwargs: Keyword arguments for `matplotlib.axes.Axes.plot`.
        """
        median = numpy.median(numpy.nan_to_num(histories, nan=numpy.inf), 0)
        # Skip infinite values to support the ``markevery`` option.
        first_index = next(
            (index for index, value in enumerate(median) if numpy.isfinite(value)),
            histories.shape[1],
        )
        axes.plot(
            range(first_index + 1, histories.shape[1] + 1),
            median[first_index:],
            **plot_kwargs,
        )
