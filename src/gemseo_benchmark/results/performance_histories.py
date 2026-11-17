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

import numpy
from matplotlib.ticker import MaxNLocator

from gemseo_benchmark.report.axis_data import AxisData
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    import datetime
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from gemseo_benchmark.results.history_item import HistoryItem


class PerformanceHistories(collections.abc.MutableSequence):
    """A collection of performance histories."""

    __histories: list[PerformanceHistory]
    """The performance histories of the collection."""

    def __init__(self, *histories: PerformanceHistory) -> None:
        """
        Args:
            *histories: The performance histories.
        """  # ruff: ignore[missing-blank-line-after-summary, multi-line-summary-first-line, missing-terminal-punctuation]
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
            history.extend(self.maximum_size) for history in self
        ])

    @property
    def maximum_size(self) -> int:
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
            for items in zip(
                *[history.items for history in self.get_equal_size_histories()],
                strict=False,
            )
        ]
        return history

    def cumulate_minimum(self) -> PerformanceHistories:
        """Return the histories of the minimum."""
        return PerformanceHistories(*[
            history.compute_cumulated_minimum() for history in self
        ])

    def plot_performance_measure_distribution(
        self,
        axes: Axes,
        extremal_feasible_performance: float | None,
        infeasible_performance_measure: float,
        plot_all_histories: bool = False,
        performance_measure_is_minimized: bool = True,
        use_elapsed_time: bool = False,
    ) -> None:
        """Plot the distribution of the performance measure.

        Args:
            axes: The axes of the plot.
            extremal_feasible_performance: The extremal feasible performance measure.
            infeasible_performance_measure: The value to replace the performance measure
                of infeasible history items when computing statistics.
            plot_all_histories: Whether to plot all the performance histories.
            performance_measure_is_minimized: Whether the performance measure
                is minimized (rather than maximized).
            use_elapsed_time: Whether to plot the distribution
                relative to the elapsed time
                (rather than relative to the number of functions evaluations).
        """
        histories = (
            self.spread_over_time(None)
            if use_elapsed_time
            else self.get_equal_size_histories()
        )
        self.__plot_distribution(
            numpy.array([
                [
                    item.performance_measure
                    if item.is_feasible
                    else infeasible_performance_measure
                    for item in history
                ]
                for history in histories
            ]),
            axes,
            "Performance measure",
            extremal_feasible_performance,
            plot_all_histories,
            performance_measure_is_minimized,
            self.__get_elapsed_seconds(use_elapsed_time),
        )

    def plot_infeasibility_measure_distribution(
        self,
        axes: Axes,
        plot_all_histories: bool = False,
        use_elapsed_time: bool = False,
    ) -> None:
        """Plot the distribution of the infeasibility measure.

        Args:
            axes: The axes of the plot.
            plot_all_histories: Whether to plot all the performance histories.
            use_elapsed_time: Whether to plot the distribution
                relative to the elapsed time
                (rather than relative to the number of functions evaluations).
        """
        histories = self.spread_over_time(None) if use_elapsed_time else self
        self.__plot_distribution(
            numpy.array([history.infeasibility_measures for history in histories]),
            axes,
            "Infeasibility measure",
            plot_all_histories=plot_all_histories,
            elapsed_times=self.__get_elapsed_seconds(use_elapsed_time),
        )

    def plot_number_of_unsatisfied_constraints_distribution(
        self,
        axes: Axes,
        plot_all_histories: bool = False,
        use_elapsed_time: bool = False,
    ) -> None:
        """Plot the distribution of the number of unsatisfied constraints.

        Args:
            axes: The axes of the plot.
            plot_all_histories: Whether to plot all the performance histories.
            use_elapsed_time: Whether to plot the distribution
                relative to the elapsed time
                (rather than relative to the number of functions evaluations).
        """
        histories = self.spread_over_time(None) if use_elapsed_time else self
        self.__plot_distribution(
            numpy.array([
                [
                    numpy.nan if n is None else n
                    for n in history.n_unsatisfied_constraints
                ]
                for history in histories
            ]),
            axes,
            "Number of unsatisfied constraints",
            plot_all_histories=plot_all_histories,
            elapsed_times=self.__get_elapsed_seconds(use_elapsed_time),
        )

    def __get_elapsed_seconds(self, use_elapsed_time: bool) -> list[float]:
        """Return the sorted elapsed times of all the history items in seconds.

        Args:
            use_elapsed_time: Whether the distribution is plotted
                relative to the elapsed time.

        Returns:
            The sorted elapsed times in seconds,
            or an empty list if the distribution is not plotted
            relative to the elapsed time.
        """
        if not use_elapsed_time:
            return []

        return [time.total_seconds() for time in self.get_elapsed_times()]

    @staticmethod
    def __plot_distribution(
        histories: numpy.ndarray,
        axes: Axes,
        y_label: str,
        infinity: float | None = None,
        plot_all_histories: bool = False,
        performance_measure_is_minimized: bool = True,
        elapsed_times: Sequence[float] = (),
    ) -> None:
        """Plot the distribution of histories data.

        Args:
            histories: The histories data.
            axes: The axes of the plot.
            y_label: The label for the vertical axis.
            infinity: The substitute value for infinite ordinates.
            plot_all_histories: Whether to plot all the performance histories.
            performance_measure_is_minimized: Whether the performance measure
                is minimized (rather than maximized).
            elapsed_times: The elapsed times in seconds
                to be used as abscissas.
                If empty, use the numbers of functions evaluations.
        """
        if elapsed_times:
            abscissas = elapsed_times
            drawstyle = "steps-post"
            step = "post"
        else:
            abscissas = range(1, histories.shape[1] + 1)
            drawstyle = "default"
            step = None

        legend_handles_offset = 0
        if plot_all_histories:
            legend_handles_offset = histories.shape[0] - 1
            axes.plot(
                abscissas,
                histories.T,
                color="black",
                label="histories",
                linestyle=":",
                drawstyle=drawstyle,
            )

        PerformanceHistories.plot_centiles_range(
            histories,
            axes,
            (0, 100),
            {"color": "lightgray", "label": "0th-100th centiles", "step": step},
            infinity,
            performance_measure_is_minimized,
            elapsed_times,
        )
        PerformanceHistories.plot_centiles_range(
            histories,
            axes,
            (25, 75),
            {"color": "gray", "label": "25th-75th centiles", "step": step},
            infinity,
            performance_measure_is_minimized,
            elapsed_times,
        )
        PerformanceHistories.plot_median(
            histories,
            axes,
            {"color": "black", "label": "median", "drawstyle": drawstyle},
            performance_measure_is_minimized,
            elapsed_times,
        )
        axes.plot(
            abscissas,
            numpy.mean(histories, 0),
            color="orange",
            label="mean",
            linestyle=":",
            drawstyle=drawstyle,
        )
        # Reorder the legend
        axes.legend(
            *zip(
                *[
                    list(zip(*axes.get_legend_handles_labels(), strict=False))[
                        index + legend_handles_offset
                    ]
                    for index in range(3 + plot_all_histories, -1, -1)
                ],
                strict=False,
            )
        )
        if elapsed_times:
            axes.set_xlabel("Elapsed time")
            axes.xaxis.set_major_formatter(AxisData.time_tick_formatter)
        else:
            axes.set_xlabel("Number of functions evaluations")

        axes.set_ylabel(y_label)

    @staticmethod
    def plot_centiles_range(
        histories: numpy.ndarray,
        axes: Axes,
        centile_range: tuple[float, float],
        fill_between_settings: Mapping[str, str],
        infinity: float | None,
        performance_measure_is_minimized: bool,
        abscissas: Sequence[float] = (),
    ) -> None:
        """Plot a range of centiles of histories data.

        Args:
            histories: The histories data.
            axes: The axes of the plot.
            centile_range: The range of centiles to be drawn.
            fill_between_settings: Keyword arguments
                for `matplotlib.axes.Axes.fill_between`.
            infinity: The substitute value for infinite ordinates.
            performance_measure_is_minimized: Whether the performance measure
                is minimized (rather than maximized).
            abscissas: The abscissas of the histories data.
                If empty, use the numbers of functions evaluations.
        """
        use_evaluations = len(abscissas) == 0
        if use_evaluations:
            abscissas = range(1, histories.shape[1] + 1)

        method = "inverted_cdf"  # N.B. This method supports infinite values.
        histories = numpy.nan_to_num(
            histories,
            nan=float("inf") if performance_measure_is_minimized else -float("inf"),
        )
        lower_centile = numpy.percentile(
            histories, min(centile_range), 0, method=method
        )
        upper_centile = numpy.percentile(
            histories, max(centile_range), 0, method=method
        )
        # Determine the first index with a finite value to plot.
        centile = lower_centile if performance_measure_is_minimized else upper_centile
        first_index = next(
            (i for i, value in enumerate(centile) if numpy.isfinite(value)),
            len(centile),
        )
        axes.plot(  # hack to get same limits/ticks
            abscissas[:first_index],
            numpy.full(
                first_index,
                centile[first_index] if first_index < len(centile) else numpy.nan,
            ),
            alpha=0,
        )

        if infinity is not None:
            if performance_measure_is_minimized:
                upper_centile = numpy.nan_to_num(upper_centile, posinf=infinity)
            else:
                lower_centile = numpy.nan_to_num(lower_centile, neginf=infinity)

        axes.fill_between(
            abscissas[first_index:],
            lower_centile[first_index:],
            upper_centile[first_index:],
            **fill_between_settings,
        )
        if use_evaluations:
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))

    @staticmethod
    def plot_median(
        histories: numpy.ndarray,
        axes: Axes,
        plot_settings: Mapping[str, str | int | float],
        performance_measure_is_minimized: bool,
        abscissas: Sequence[float] = (),
    ) -> None:
        """Plot a range of centiles of histories data.

        Args:
            histories: The histories data.
            axes: The axes of the plot.
            plot_settings: Keyword arguments for `matplotlib.axes.Axes.plot`.
            performance_measure_is_minimized: Whether the performance measure
                is minimized (rather than maximized).
            abscissas: The abscissas of the histories data.
                If empty, use the numbers of functions evaluations.
        """
        if len(abscissas) == 0:
            abscissas = range(1, histories.shape[1] + 1)

        median = numpy.median(
            numpy.nan_to_num(
                histories,
                nan=float("inf") if performance_measure_is_minimized else -float("inf"),
            ),
            0,
        )
        # Skip infinite values to support the `markevery` option.
        first_index = next(
            (index for index, value in enumerate(median) if numpy.isfinite(value)),
            histories.shape[1],
        )
        axes.plot(
            abscissas[first_index:],
            median[first_index:],
            **plot_settings,
        )

    def switch_performance_measure_sign(self) -> None:
        """Switch the sign of the performance measure."""
        for history in self:
            history.switch_performance_measure_sign()

    def get_elapsed_times(self) -> list[datetime.timedelta]:
        """Return the sorted elapsed times of all the history items.

        Returns:
            The sorted elapsed times.
        """
        return sorted([item.elapsed_time for history in self for item in history])

    def spread_over_time(
        self, number_of_scalar_constraints: int | None
    ) -> PerformanceHistories:
        """Spread the histories over all the elapsed times.

        Args:
            number_of_scalar_constraints: The number of scalar constraints
                of the underlying problem.
                If `None`, the number of unsatisfied constraints is unknown
                before the first item of a history.

        Returns:
            The performance histories with has as many items as total elapsed times.
        """
        timeline = self.get_elapsed_times()
        return self.__class__(*[
            history.spread_over_timeline(timeline, number_of_scalar_constraints)
            for history in self
        ])

    def get_numbers_of_discipline_executions(self) -> list[int]:
        """Return the sorted numbers of discipline executions of all the history items.

        Returns:
            The sorted numbers of discipline executions.
        """
        return sorted({
            item.number_of_discipline_executions for history in self for item in history
        })

    def spread_over_numbers_of_discipline_executions(
        self, number_of_scalar_constraints: int
    ) -> PerformanceHistories:
        """Spread the histories over all the numbers of discipline executions.

        Args:
            number_of_scalar_constraints: The number of scalar constraints
                of the underlying problem.

        Returns:
            The performance histories with has as many items as total numbers
            of discipline executions.
        """
        numbers_of_discipline_executions = self.get_numbers_of_discipline_executions()
        return self.__class__(*[
            history.spread_over_numbers_of_discipline_executions(
                numbers_of_discipline_executions, number_of_scalar_constraints
            )
            for history in self
        ])
