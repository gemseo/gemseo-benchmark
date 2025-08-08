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
"""Plotting the range of performance history data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import matplotlib

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.report.report_plot import ReportPlot

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )
    from gemseo_benchmark.report.axis_data import AbscissaData
    from gemseo_benchmark.report.axis_data import OrdinateData
    from gemseo_benchmark.results.performance_histories import PerformanceHistories


class RangePlot(ReportPlot):
    """A plot of the range of performance history data."""

    def plot(
        self,
        axes: matplotlib.axes.Axes,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        ordinate_data: OrdinateData,
        abscissa_data: AbscissaData,
        infinity_replacement: float,
        plot_only_median: bool,
        plot_all_histories: bool,
        data_is_minimized: bool,
        nan_replacement: float,
        use_ordinate_log_scale: bool,
    ) -> None:
        """Make the plot.

        Args:
            axes: The axes of the plot.
            performance_histories: The performance histories
                of algorithm configurations.
            ordinate_data: The data of the ordinate axis.
            abscissa_data: The data of the abscissa axis.
            infinity_replacement: The finite substitute value for infinite ordinates
                to enable `matplotlib.axes.Axes.fill_between`.
            plot_only_median: Whether to plot only the median and no other centile.
            plot_all_histories: Whether to plot all the performance histories.
            data_is_minimized: Whether the data is minimized (rather than maximized).
            nan_replacement: The value to replace NaN history entries
                to enable the computation of centiles.
            use_ordinate_log_scale: Whether to use a logarithmic scale
                for the ordinate axis.
        """
        data_minimum = float("inf")
        for algorithm_configuration, histories in performance_histories.items():
            name = algorithm_configuration.name
            data = ordinate_data.get(abscissa_data.spread(histories))
            data_minimum = min(data.min(), data_minimum)
            abscissas = abscissa_data.get(histories)
            if plot_all_histories:
                axes.plot(
                    abscissas,
                    data.T,
                    color=self._plot_kwargs[name]["color"],
                    linestyle=self._HISTORY_LINESTYLE,
                    drawstyle=self._DRAWSTYLE,
                )

            if not plot_only_median:
                self._plot_centiles_range(
                    axes,
                    abscissas,
                    data,
                    (0, 100),
                    {
                        "alpha": self._alpha,
                        "color": self._plot_kwargs[name]["color"],
                    },
                    infinity_replacement,
                    data_is_minimized,
                    nan_replacement,
                )

            self._plot_median(
                axes,
                abscissas,
                data,
                dict(drawstyle=self._DRAWSTYLE, **self._plot_kwargs[name]),
                nan_replacement,
            )

        axes.set_xbound(abscissas.min(), abscissas.max())
        axes.set_ymargin(0.01)
        if use_ordinate_log_scale:
            scale = (
                matplotlib.scale.SymmetricalLogScale
                if data_minimum < 0
                else matplotlib.scale.LogScale
            )
            axes.set_yscale(scale.name)

        axes.grid(**self._grid_kwargs)
        axes.legend()
