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
"""Computation of target values out of algorithms performance histories.

Consider a problem to be solved by an iterative algorithm, e.g. an optimization problem
or a root-finding problem. Targets are values, i.e. values of the objective function or
values of the residual norm, ranging from a first acceptable value to the best known
value for the problem. Targets are used to estimate the efficiency (relative to the
number of problem functions evaluations) of an algorithm to solve a problem (or several)
and computes its data profile (see :mod:`.data_profiles.data_profile`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from gemseo.utils.matplotlib_figure import save_show_figure
from numpy import array
from numpy import linspace
from numpy import logical_not

from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure


class TargetValues(PerformanceHistory):
    """Target values of a problem."""

    def compute_target_hits_history(
        self, values_history: PerformanceHistory
    ) -> list[int]:
        """Compute the history of the number of target hits for a performance history.

        Args:
            values_history: The history of values.

        Returns:
            The history of the number of target hits.
        """
        minimum_history = values_history.compute_cumulated_minimum()
        return [
            [minimum <= target for target in self].count(True)
            for minimum in minimum_history
        ]

    def plot(self, show: bool = True, file_path: str | Path | None = None) -> Figure:
        """Plot the target values.

        Args:
            show: Whether to show the plot.
            file_path: The path where to save the plot.
                If ``None``, the plot is not saved.

        Returns:
            A figure showing the target values.
        """
        targets_number = len(self)
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.set_title("Target values")
        plt.xlabel("Target index")
        plt.xlim([0, targets_number + 1])
        plt.xticks(linspace(1, targets_number, dtype=int))
        plt.ylabel("Target value")
        indexes, history_items = self.get_plot_data()

        # Plot the feasible target values
        objective_values = [item.objective_value for item in history_items]
        is_feasible = array([item.is_feasible for item in history_items])
        if is_feasible.any():
            axes.plot(
                array(indexes)[is_feasible],
                array(objective_values)[is_feasible],
                color="black",
                marker="o",
                linestyle="",
                label="feasible",
            )

        # Plot the infeasible target values
        is_infeasible = logical_not(is_feasible)
        if is_infeasible.any():
            axes.plot(
                array(indexes)[is_infeasible],
                array(objective_values)[is_infeasible],
                color="red",
                marker="x",
                linestyle="",
                label="infeasible",
            )

        plt.legend()

        save_show_figure(fig, show, file_path)

        return fig
