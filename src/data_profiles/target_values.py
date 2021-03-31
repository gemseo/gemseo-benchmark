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
"""Computation of target values out of optimization histories"""
from functools import reduce
from itertools import chain, repeat
from typing import List, Optional, Tuple

from matplotlib.pyplot import (figure, semilogy, show, xlabel, xlim, xticks, ylabel)
from numpy import array, inf, linspace, ndarray


class TargetValues(object):
    """Compute target values for an objective to minimize.

    Attributes:
        _values_histories: The histories of objective values.
        _measures_histories: The histories of infeasibility measures.

    """

    def __init__(self):
        self._values_histories = list()
        self._measures_histories = list()

    def add_history(
            self,
            values_history,  # type: List
            measures_history=None,  # type: Optional[List]
            feasibility_history=None,  # type: Optional[List[bool]]
    ):
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
        self._values_histories.append(values_history)
        if measures_history is None and feasibility_history is not None:
            if len(feasibility_history) != len(values_history):
                raise ValueError("Values history and feasibility history must have "
                                 "same length.")
            measures_history = [0.0 if a_feas else inf
                                for a_feas in feasibility_history]
        elif measures_history is None:
            measures_history = [0.0] * len(values_history)
        if len(measures_history) != len(values_history):
            raise ValueError("Values history and measures history must have same "
                             "length.")
        self._measures_histories.append(measures_history)

    def compute(
            self,
            targets_number,  # type: int
            budget_min=1,  # type: Optional[int]
            plot=False  # type: bool
    ):  # type: (...) -> ndarray
        """Compute target values for a function from histories of its values.

        Args:
            targets_number: The number of targets to compute.
            budget_min: The evaluation budget to be used to define the easiest target.
            plot: Whether to plot the target values.

        Returns:
            The target values of the function.

        """
        # Build the coupled histories of objective values and infeasibility measures
        full_histories = [list(zip(a_val_hist, a_meas_hist)) for a_val_hist, a_meas_hist
                          in zip(self._values_histories, self._measures_histories)]

        # Extend the histories to a common size by repeating their last value
        maximal_size = max(len(a_history) for a_history in full_histories)
        full_histories = [
            list(chain(a_hist, repeat(a_hist[-1], (maximal_size - len(a_hist)))))
            for a_hist in full_histories
        ]

        # Compute the history of the minimum value
        minima_histories = [[reduce(TargetValues._min, a_full_hist[:i])
                             for i in range(1, len(a_full_hist) + 1)]
                            for a_full_hist in full_histories]
        minima_histories = array(minima_histories)
        minimum_history = minima_histories.mean(axis=0)

        # Compute a budget scale
        budget_max = len(minimum_history)
        budget_scale = TargetValues._compute_budget_scale(budget_min, budget_max,
                                                          targets_number)

        # Compute the target values
        target_values = minimum_history[budget_scale - 1, :]

        # Plot the target values
        if plot:
            objective_values = [inf if a_meas > 0.0 else a_value
                                for a_value, a_meas in target_values]
            self._plot(objective_values)

        return target_values

    @staticmethod
    def _plot(
            objective_target_values  # type: List
    ):

        """Compute and plot the target values.

            Args:
                objective_target_values: The objective target values.

        """
        targets_number = len(objective_target_values)
        figure()
        xlabel("Target index")
        xlim([0, targets_number + 1])
        xticks(linspace(1, targets_number, dtype=int))
        ylabel("Target value")
        semilogy(range(1, targets_number + 1), objective_target_values,
                 marker="o", linestyle="")
        show()

    @staticmethod
    def _min(
            hist_item_1,  # type: Tuple
            hist_item_2  # type: Tuple
    ):  # type:(...)-> Tuple
        """Return the smallest of two history items.

        Args:
            hist_item_1: A history item.
            hist_item_2: Another history item.

        Returns:
            The smallest of the two history items.

        """
        value_1, meas_1 = hist_item_1
        value_2, meas_2 = hist_item_2
        if 0 < meas_1 <= meas_2:
            return hist_item_1
        elif meas_1 > 0:
            return hist_item_2
        elif meas_2 > 0:
            return hist_item_1
        elif value_1 <= value_2:
            return hist_item_1
        else:
            return hist_item_2

    @staticmethod
    def _compute_budget_scale(
            budget_min,  # type: int
            budget_max,  # type: int
            budgets_number  # type: int
    ):  # type: (...) -> ndarray
        """Compute a scale of evaluation budgets, whose progression relates to
        complexity in terms of evaluation cost.

        N.B. here the evaluation cost is assumed linear with respect to the number of
        evaluations.

        Args:
            budget_min: The minimum number of evaluations
            budget_max: The maximum number of evaluations
            budgets_number: the number of budgets

        Returns:
            distribution of evaluation budgets

        """
        budget_scale = linspace(budget_min, budget_max, budgets_number, dtype=int)
        return budget_scale
