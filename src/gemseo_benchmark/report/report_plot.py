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
"""Plotting performance history data."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar

import numpy
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    import matplotlib

    from gemseo_benchmark import ConfigurationPlotOptions


class ReportPlot(metaclass=ABCGoogleDocstringInheritanceMeta):
    """A plot of performance history data."""

    _DRAWSTYLE: ClassVar[str] = "steps-post"
    """The 'drawstyle' argument value for `matplotlib.axes.Axes.plot`."""

    _HISTORY_LINESTYLE: ClassVar[str] = ":"
    """The 'linestyle' argument value for `matplotlib.axes.Axes.plot`
    when plotting individual histories."""

    def __init__(
        self,
        plot_kwargs: Mapping[str, ConfigurationPlotOptions],
        grid_kwargs: Mapping[str, str],
        alpha: float,
        matplotlib_log_scale: str,
    ) -> None:
        """
        Args:
            plot_settings: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.
            grid_settings: The keyword arguments of `matplotlib.pyplot.grid`.
            alpha: The opacity level for overlapping areas.
                (Refer to the Matplotlib documentation.)
            matplotlib_log_scale: The Matplotlib value for logarithmic scale.
            time_formatter: The formatter for time tick labels.
        """  # noqa: D205, D212
        self._alpha = alpha
        self._grid_kwargs = grid_kwargs
        self._matplotlib_log_scale = matplotlib_log_scale
        self._plot_kwargs = plot_kwargs

    @abstractmethod
    def plot(self) -> None:
        """Make the plot."""

    @staticmethod
    def _plot_median(
        axes: matplotlib.axes.Axes,
        abscissas: Sequence[int | float],
        data: numpy.ndarray,
        plot_kwargs: Mapping[str, str | int | float],
        nan_replacement: float,
    ) -> None:
        """Plot a range of centiles of histories data.

        Args:
            axes: The axes of the plot.
            abscissas: The abscissas of the plot.
            data: The histories data.
            plot_kwargs: Keyword arguments for `matplotlib.axes.Axes.plot`.
            nan_replacement: The value to replace NaN history entries
                to enable the computation of centiles.
        """
        median = numpy.median(numpy.nan_to_num(data, nan=nan_replacement), 0)
        # Skip infinite values to support the ``markevery`` option.
        first_index = next(
            (index for index, value in enumerate(median) if numpy.isfinite(value)),
            data.shape[1],
        )
        axes.plot(abscissas[first_index:], median[first_index:], **plot_kwargs)

    @staticmethod
    def _plot_centiles_range(
        axes: matplotlib.axes.Axes,
        abscissas: Sequence[int | float],
        data: numpy.ndarray,
        centile_range: tuple[float, float],
        fill_between_kwargs: Mapping[str, str | float],
        infinity_replacement: float,
        data_is_minimized: bool,
        nan_replacement: float,
    ) -> None:
        """Plot a range of centiles of histories data.

        Args:
            axes: The axes of the plot.
            abscissas: The abscissas of the plot.
            data: The histories data.
            centile_range: The range of centiles.
            fill_between_kwargs: Keyword arguments
                for `matplotlib.axes.Axes.fill_between`.
            infinity_replacement: The finite substitute value for infinite ordinates
                to enable `matplotlib.axes.Axes.fill_between`.
            data_is_minimized: Whether the data is minimized (rather than maximized).
            nan_replacement: The value to replace NaN history entries
                to enable the computation of centiles.
        """
        method = "inverted_cdf"  # N.B. This method supports infinite values.
        data = numpy.nan_to_num(data, nan=nan_replacement)
        lower_centile = numpy.percentile(data, min(centile_range), 0, method=method)
        upper_centile = numpy.percentile(data, max(centile_range), 0, method=method)
        # Determine the first index with a finite value to plot.
        centile = lower_centile if data_is_minimized else upper_centile
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
        if data_is_minimized:
            upper_centile = numpy.nan_to_num(upper_centile, posinf=infinity_replacement)
        else:
            lower_centile = numpy.nan_to_num(lower_centile, neginf=infinity_replacement)

        axes.fill_between(
            abscissas[first_index:],
            lower_centile[first_index:],
            upper_centile[first_index:],
            edgecolor="none",
            step="post",
            **fill_between_kwargs,
        )
