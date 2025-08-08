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
"""Getting data for a plot axis."""

from __future__ import annotations

import datetime
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar

import matplotlib
import matplotlib.axes
import numpy
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

from gemseo_benchmark.results.performance_histories import PerformanceHistories

if TYPE_CHECKING:
    from gemseo.typing import IntegerArray
    from gemseo.typing import RealArray

    from gemseo_benchmark.results.history_item import HistoryItem
    from gemseo_benchmark.results.performance_histories import PerformanceHistories


class AxisData(metaclass=ABCGoogleDocstringInheritanceMeta):
    """The data of a plot axis."""

    time_tick_formatter: ClassVar[matplotlib.ticker.FuncFormatter] = (
        matplotlib.ticker.FuncFormatter(lambda x, _: str(datetime.timedelta(seconds=x)))
    )
    """The formatter for time tick labels."""

    def __init__(self, axes: matplotlib.axes.Axes) -> None:
        """
        Args:
            axes: The axes of the plot.
        """  # noqa: D205, D212
        self._axes = axes

    @property
    @abstractmethod
    def _label(self) -> str:
        """The label of the axis."""

    @abstractmethod
    def get(
        self, performance_histories: PerformanceHistories
    ) -> IntegerArray | RealArray:
        """Return the axis data associated with performance histories.

        Args:
            performance_histories: The performance histories.

        Returns:
            The axis data.
        """

    @abstractmethod
    def _format_linear_integer_ticks(self) -> None:
        """Format linearly scaled ticks."""

    @abstractmethod
    def _format_linear_time_ticks(self) -> None:
        """Format linearly scaled time ticks."""


class OrdinateData(AxisData):
    """The data of an ordinate axis."""

    def __init__(self, axes: matplotlib.axes.Axes) -> None:  # noqa:D107
        super().__init__(axes)
        self._axes.set_ylabel(self._label)

    def get(  # noqa: D102
        self, performance_histories: PerformanceHistories
    ) -> IntegerArray | RealArray:
        return numpy.array([
            [self._get_ordinate(history_item) for history_item in performance_history]
            for performance_history in performance_histories
        ])

    @abstractmethod
    def _get_ordinate(self, history_item: HistoryItem, *args) -> float:
        """Return the ordinate associated with a performance history item.

        Args:
            history_item: The performance history item.

        Returns:
            The ordinate.
        """

    def _format_linear_integer_ticks(self) -> None:
        """Format the linearly scaled ticks."""
        self._axes.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    def _format_linear_time_ticks(self) -> None:
        """Format linearly scaled time ticks."""
        self._axes.yaxis.set_major_formatter(self.time_tick_formatter)


class PerformanceData(OrdinateData):
    """The performance data of an ordinate axis."""

    def __init__(
        self,
        axes: matplotlib.axes.Axes,
        infeasible_performance_measure: float,
        label: str,
    ) -> None:
        """
        Args:
            infeasible_performance_measure: The performance measure
                for infeasible history items.
        """  # noqa: D205, D212
        self.__label = label
        super().__init__(axes)
        self.__infeasible_performance_mesasure = infeasible_performance_measure

    @property
    def _label(self) -> str:  # noqa: D102
        return self.__label

    def _get_ordinate(self, history_item: HistoryItem) -> float:
        return (
            history_item.performance_measure
            if history_item.is_feasible
            else self.__infeasible_performance_mesasure
        )


class InfeasibilityData(OrdinateData):
    """The infeasibility data of an ordinate axis."""

    _label: ClassVar[str] = "Infeasibility measure"

    def _get_ordinate(self, history_item: HistoryItem) -> float:
        return history_item.infeasibility_measure


class ConstraintData(OrdinateData):
    """The constraint unsatisfaction data of an ordinate axis."""

    _label: ClassVar[str] = "Number of unsatisfied constraints"

    def __init__(self, axes: matplotlib.axes.Axes) -> None:  # noqa: D107
        super().__init__(axes)
        self._format_linear_integer_ticks()

    def _get_ordinate(self, history_item: HistoryItem) -> float:
        number = history_item.n_unsatisfied_constraints
        return numpy.nan if number is None else number


class TimeOrdinateData(OrdinateData):
    """The elapsed time data of an ordinate axis."""

    _label: ClassVar[str] = "Elapsed time"

    def __init__(self, axes: matplotlib.axes.Axes) -> None:  # noqa:D107
        super().__init__(axes)
        self._format_linear_time_ticks()

    def _get_ordinate(self, history_item: HistoryItem) -> float:
        return history_item.elapsed_time.total_seconds()


class AbscissaData(AxisData):
    """The data of an abscissa axis."""

    def __init__(
        self,
        axes: matplotlib.axes.Axes,
        number_of_scalar_constraints: int,
        use_log_scale: bool,
    ) -> None:
        """
        Args:
            number_of_scalar_constraints: The number of scalar constraints
                of the underlying problem.
            use_log_scale: Whether to use a logarithmic scale for the axis.
        """  # noqa: D205, D212
        super().__init__(axes)
        self._axes.set_xlabel(self._label)
        self._axes.tick_params(axis="x", labelrotation=90)
        self._number_of_scalar_constraints = number_of_scalar_constraints
        if use_log_scale:
            self._axes.set_xscale(matplotlib.scale.LogScale.name)
        else:
            self._format_linear_ticks()

    @abstractmethod
    def spread(
        self, performance_histories: PerformanceHistories
    ) -> PerformanceHistories:
        """Spread histories."""

    @abstractmethod
    def _format_linear_ticks(self) -> None:
        """Format the linearly scaled ticks."""

    def _format_linear_integer_ticks(self) -> None:
        """Format the linearly scaled ticks."""
        self._axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    def _format_linear_time_ticks(self) -> None:
        """Format linearly scaled time ticks."""
        self._axes.xaxis.set_major_formatter(self.time_tick_formatter)


class DisciplineData(AbscissaData):
    """The discipline data of an abscissa axis."""

    _label: ClassVar[str] = "Number of discipline executions"

    def _format_linear_ticks(self) -> None:
        self._format_linear_integer_ticks()

    def get(self, performance_histories: PerformanceHistories) -> IntegerArray:  # noqa: D102
        return numpy.array(performance_histories.get_numbers_of_discipline_executions())

    def spread(  # noqa: D102
        self, performance_histories: PerformanceHistories
    ) -> PerformanceHistories:
        return performance_histories.spread_over_numbers_of_discipline_executions(
            self._number_of_scalar_constraints
        )


class IterationData(AbscissaData):
    """The iteration data of an abscissa axis."""

    _label: ClassVar[str] = "Number of iterations"

    def _format_linear_ticks(self) -> None:
        self._format_linear_integer_ticks()

    def get(self, performance_histories: PerformanceHistories) -> IntegerArray:  # noqa: D102
        return numpy.arange(1, performance_histories.maximum_size + 1)

    def spread(  # noqa: D102
        self, performance_histories: PerformanceHistories
    ) -> PerformanceHistories:
        return performance_histories.get_equal_size_histories()


class TimeAbscissaData(AbscissaData):
    """The elapsed time data of an abscissa axis."""

    _label: ClassVar[str] = "Elapsed time"

    def _format_linear_ticks(self) -> None:
        self._format_linear_time_ticks()

    def get(self, performance_histories: PerformanceHistories) -> RealArray:  # noqa: D102
        return numpy.array([
            time.total_seconds() for time in performance_histories.get_elapsed_times()
        ])

    def spread(  # noqa: D102
        self, performance_histories: PerformanceHistories
    ) -> PerformanceHistories:
        return performance_histories.spread_over_time(
            self._number_of_scalar_constraints
        )
