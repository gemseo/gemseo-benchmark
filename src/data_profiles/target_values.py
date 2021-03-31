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

from numpy import array, inf, linspace, ndarray


class TargetValues(object):
    """Compute target values for an objective to minimize."""

    @staticmethod
    def compute_target_values(
            values_histories,  # type: Iterable[List]
            targets_number,  # type: int
            budget_min=1,  # type: Optional[int]
            feasibility_histories=None,  # type: Optional[Iterable[List[bool]]]
            measures_histories=None  # type: Optional[Iterable[List]]
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
            measures_histories:  The histories of infeasibility measures.
                If None then the behavior depends on the feasibility_histories
                argument; otherwise the feasibility_histories argument is not
                considered.

        Returns:
            The target values of the function.

        """
        # Check the histories
        values_histories, measures_histories = TargetValues._check_histories(
            values_histories, feasibility_histories, measures_histories
        )
        full_histories = [list(zip(a_val_hist, a_meas_hist)) for a_val_hist, a_meas_hist
                          in zip(values_histories, measures_histories)]

        # Extend the histories to a common size by repeating their last value
        maximal_size = max(len(a_history) for a_history in full_histories)
        full_histories = [
            list(chain(a_hist, repeat(a_hist[-1], (maximal_size - len(a_hist)))))
            for a_hist in full_histories
        ]

        # Compute the history of the minimum value
        minima_histories = [[reduce(TargetValues._order, a_full_hist[:i])
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

        return target_values

    @staticmethod
    def _check_histories(
            values_histories,  # type: Iterable[List]
            feasibility_histories=None,  # type: Optional[Iterable[List[bool]]]
            measures_histories=None  # type: Optional[Iterable[List]]
    ):  # type: (...) -> Tuple[Iterable[List], Iterable[List]]
        """Set the default measures of constraints violation if they are not
        provided, and check the length of the histories.

       Args:
            values_histories: The histories of the function values.
                N.B. in a history the value at index i is assumed to have been obtained
                with i+1 evaluations.
            feasibility_histories: The histories of the solutions feasibility
                If None then all solutions are assumed feasible.
            measures_histories:  The histories of infeasibility measures.
                If None then the behavior depends on the feasibility_histories
                argument; otherwise the feasibility_histories argument is not
                considered.

        Returns:
            (The values histories, The infeasibility measures histories)

        """
        if measures_histories is None and feasibility_histories is not None:
            measures_histories = [[0.0 if a_feas else inf for a_feas in a_history]
                                  for a_history in feasibility_histories]
        elif measures_histories is None:
            measures_histories = [[0.0] * len(a_history) for a_history in
                                  values_histories]
        for a_val_hist, a_viol_hist in zip(values_histories, measures_histories):
            if len(a_val_hist) != len(a_viol_hist):
                raise ValueError("Histories of values and feasibility must have same "
                                 "size")
        return values_histories, measures_histories

    @staticmethod
    def _order(a_hist_item, other_hist_item):
        # TODO: docstring
        a_value, a_meas = a_hist_item
        other_value, other_meas = other_hist_item

        if a_meas > 0 and other_meas >= a_meas:
            return a_hist_item
        elif a_meas > 0:
            return other_hist_item
        elif other_meas > 0:
            return a_hist_item
        elif a_value <= other_value:
            return a_hist_item
        else:
            return other_hist_item

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
