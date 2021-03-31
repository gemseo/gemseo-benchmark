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
from typing import Iterable, List, Optional, Tuple

from numpy import array, inf, linspace, ndarray, vstack


class TargetValues(object):
    """Compute target values for an objective to minimize."""

    @staticmethod
    def compute_target_values(
            values_histories,  # type: Iterable[List]
            targets_number,  # type: int
            budget_min=1,  # type: Optional[int]
            feasibility_histories=None,  # type: Optional[Iterable[List[bool]]]
    ):  # type: (...) -> ndarray
        """Compute target values for a function from histories of its values.

        Args:
            values_histories: The histories of the function values.
                N.B. in a history the value at index i is assumed to have been obtained
                with i+1 evaluations.
            targets_number: The number of targets to compute.
            budget_min: The evaluation budget to be used to define the easiest target.
            feasibility_histories: The histories of the solutions feasibility
                If None then all solutions are assumed feasible.

        Returns:
            The target values of the function.

        """
        # Check the histories lengths
        if feasibility_histories is None:
            feasibility_histories = [[True] * len(a_history) for a_history
                                     in values_histories]
        for a_val_hist, a_feas_hist in zip(values_histories, feasibility_histories):
            if len(a_val_hist) != len(a_feas_hist):
                raise ValueError("Histories of values and feasibility must have same "
                                 "size")

        # Take feasibility into account
        print(values_histories)
        values_histories = [[a_value if a_feas else inf
                             for a_value, a_feas in zip(a_val_hist, a_feas_hist)]
                            for a_val_hist, a_feas_hist in zip(values_histories,
                                                               feasibility_histories)]
        print(values_histories)
        # TODO: use a feasibility measure / penalty

        # Extend the histories to a common size by repeating their last value
        maximal_size = max(len(a_history) for a_history in values_histories)
        values_histories = [
            list(chain(a_hist, repeat(a_hist[-1], (maximal_size - len(a_hist)))))
            for a_hist in values_histories
        ]

        # Compute the history of the minimum value
        minimum_history = vstack([minimum.accumulate(a_hist) for a_hist in
                                  values_histories])
        minimum_history = minimum_history.mean(axis=0)

        # Compute a budget scale
        budget_max = len(minimum_history)
        budget_scale = TargetValues._compute_budget_scale(budget_min, budget_max,
                                                          targets_number)

        # Compute the target values
        target_values = minimum_history[budget_scale - 1]

        return target_values

    @staticmethod
    def _compute_budget_scale(budget_min,  # type: int
                              budget_max,  # type: int
                              budgets_number  # type: int
                              ):  # type: (...) -> ndarray

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
