# -*- coding: utf-8 -*-
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
"""Generation of targets for a problem to be solved by an iterative algorithm.

The targets are generated out of algorithms histories considered to be of reference:
the median of the reference histories is computed
and a uniformly distributed subset (of the required size) of this median history is
extracted.
"""
from itertools import chain, repeat
from typing import List, Optional

from matplotlib.pyplot import (figure, savefig, semilogy, show as pyplot_show, xlabel,
                               xlim, xticks, ylabel)
from numpy import inf, linspace, ndarray

from data_profiles.performance_history import PerformanceHistory
from data_profiles.target_values import TargetValues


class TargetsGenerator(object):
    """Compute target values for an objective to minimize."""

    def __init__(self):  # type: (...) -> None
        self._histories = list()

    def add_history(
            self,
            values_history,  # type: List[float]
            measures_history=None,  # type: Optional[List[float]]
            feasibility_history=None,  # type: Optional[List[bool]]
    ):  # type: (...) -> None
        """Add a history of objective values.

        Args:
            values_history: A history of objective values.
                N.B. the value at index i is assumed to have been obtained with i+1
                evaluations.
            measures_history: A history of infeasibility measures.
                If None then measures are set to zero in case of feasibility and set
                to infinity otherwise.
            feasibility_history: A history of feasibilities.
                If None then feasibility is always assumed.
        """
        history = PerformanceHistory(
            values_history, measures_history, feasibility_history
        )
        self._histories.append(history)

    def run(
            self,
            targets_number,  # type: int
            budget_min=1,  # type: int
            feasible=True,  # type: bool
            show=True,  # type: bool
            destination_path=None,  # type: Optional[str]
    ):  # type: (...) -> TargetValues
        """Compute the target values for a function from the histories of its values.

        Args:
            targets_number: The number of targets to compute.
            budget_min: The evaluation budget to be used to define the easiest target.
            plot: Whether to plot the target values.
            feasible: Whether to generate only feasible targets.
            show: Whether to show the plot.
            destination_path: The path where to save the plot.
                By default the plot is not saved.

        Returns:
            The target values of the function.
        """
        # Optionally, filter out the first infeasible items
        if feasible:
            histories = [
                hist.remove_leading_infeasible() for hist in self._histories
            ]
        else:
            histories = list(self._histories)

        # Compute the history of the minimum value
        budget_max = max(len(history) for history in histories)
        minimum_histories = list()
        for history in histories:
            min_hist = history.compute_cumulated_minimum()
            # If necessary, extend the history by repeating its last value
            if len(min_hist) < budget_max:
                min_hist.history_items = list(chain(
                    min_hist, repeat(min_hist[-1], (budget_max - len(min_hist)))
                ))
            minimum_histories.append(min_hist)
        median_history = PerformanceHistory.compute_median_history(minimum_histories)

        # Compute a budget scale
        budget_scale = TargetsGenerator._compute_budget_scale(
            budget_min, budget_max, targets_number
        )

        # Compute the target values
        target_values = TargetValues()
        target_values.history_items = [
            median_history[item - 1] for item in budget_scale
        ]

        # Plot the target values
        if show or destination_path is not None:
            objective_values = [
                inf if item.infeasibility_measure > 0.0 else item.objective_value
                for item in target_values
            ]
            self._plot(objective_values, show, destination_path)

        return target_values

    @staticmethod
    def _plot(
            objective_target_values,  # type: List[float]
            show=True,  # type: bool
            destination_path=None,  # type: Optional[str]
    ):  # type: (...) -> None
        """Compute and plot the target values.

        Args:
            objective_target_values: The objective target values.
            show: If True, show the plot.
            destination_path: The path where to save the plot.
                If None, the plot is not saved.
        """
        targets_number = len(objective_target_values)
        figure()
        xlabel("Target index")
        xlim([0, targets_number + 1])
        xticks(linspace(1, targets_number, dtype=int))
        ylabel("Target value")
        semilogy(range(1, targets_number + 1), objective_target_values,
                 marker="o", linestyle="")

        # Save and/or show the plot
        if destination_path is not None:
            savefig(destination_path)
        if show:
            pyplot_show()

    @staticmethod
    def _compute_budget_scale(
            budget_min,  # type: int
            budget_max,  # type: int
            budgets_number  # type: int
    ):  # type: (...) -> ndarray
        """Compute a scale of evaluation budgets.

         The progression of the scale relates to complexity in terms of evaluation cost.

        N.B. here the evaluation cost is assumed linear with respect to the number of
        evaluations.

        Args:
            budget_min: The minimum number of evaluations.
            budget_max: The maximum number of evaluations.
            budgets_number: The number of budgets.

        Returns:
            The distribution of evaluation budgets.
        """
        budget_scale = linspace(budget_min, budget_max, budgets_number, dtype=int)
        return budget_scale
