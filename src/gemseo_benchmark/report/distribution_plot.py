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
"""Plotting the distribution of performance history data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import numpy

from gemseo_benchmark.report.report_plot import ReportPlot

if TYPE_CHECKING:
    from gemseo_benchmark.report.axis_data import AbscissaData
    from gemseo_benchmark.report.axis_data import OrdinateData
    from gemseo_benchmark.results.performance_histories import PerformanceHistories


class DistributionPlot(ReportPlot):
    """A plot of the distribution of performance history data."""

    def plot(
        self,
        axes: matplotlib.axes.Axes,
        performance_histories: PerformanceHistories,
        ordinate_data: OrdinateData,
        abscissa_data: AbscissaData,
        infinity_replacement: float,
        plot_all_histories: bool,
        data_is_minimized: bool,
        nan_replacement: float,
        use_ordinate_log_scale: bool,
    ) -> None:
        """Make the plot.

        Args:
            axes: The axes of the plot.
            performance_histories: The performance histories
                of an algorithm configuration.
            ordinate_data: The data of the ordinate axis.
            abscissa_data: The data of the abscissa axis.
            infinity_replacement: The finite substitute value for infinite ordinates
                to enable `matplotlib.axes.Axes.fill_between`.
            plot_all_histories: Whether to plot all the performance histories.
            data_is_minimized: Whether the data is minimized (rather than maximized).
            nan_replacement: The value to replace NaN history entries
                to enable the computation of centiles.
            use_ordinate_log_scale: Whether to use a logarithmic scale
                for the ordinate axis.
        """
        data = ordinate_data.get(abscissa_data.spread(performance_histories))
        abscissas = abscissa_data.get(performance_histories)
        legend_handles_offset = 0
        if plot_all_histories:
            legend_handles_offset = data.shape[0] - 1
            axes.plot(
                abscissas,
                data.T,
                color="black",
                label="histories",
                linestyle=self._HISTORY_LINESTYLE,
                drawstyle=self._DRAWSTYLE,
            )

        for centile_range, color in (((0, 100), "lightgray"), ((25, 75), "gray")):
            self._plot_centiles_range(
                axes,
                abscissas,
                data,
                centile_range,
                {
                    "color": color,
                    "label": f"{centile_range[0]}th-{centile_range[1]}th centiles",
                },
                infinity_replacement,
                data_is_minimized,
                nan_replacement,
            )

        self._plot_median(
            axes,
            abscissas,
            data,
            {"color": "black", "label": "median", "drawstyle": self._DRAWSTYLE},
            nan_replacement,
        )
        axes.plot(
            abscissas,
            numpy.mean(data, 0),
            color="orange",
            label="mean",
            linestyle=":",
            drawstyle=self._DRAWSTYLE,
        )
        axes.set_xbound(abscissas.min(), abscissas.max())
        axes.set_ymargin(0.01)
        if use_ordinate_log_scale:
            scale = (
                matplotlib.scale.SymmetricalLogScale
                if numpy.nan_to_num(data, nan=nan_replacement).min() < 0
                else matplotlib.scale.LogScale
            )
            axes.set_yscale(scale.name)

        axes.grid(**self._grid_kwargs)
        # Order the legend from the mean to the widest range of centiles.
        handles_and_labels = list(zip(*axes.get_legend_handles_labels(), strict=True))
        axes.legend(
            *zip(
                *[
                    handles_and_labels[index + legend_handles_offset]
                    for index in range(3 + plot_all_histories, -1, -1)
                ],
                strict=True,
            )
        )
