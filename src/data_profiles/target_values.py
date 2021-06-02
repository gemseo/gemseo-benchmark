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
"""Computation of target values out of algorithms performance histories.

Consider a problem to be solved by an iterative algorithm,
e.g. an optimization problem or a root-finding problem.
Targets are values,
i.e. values of the objective function or values of the residual norm,
ranging from a first acceptable value to the best known value for the problem.
Targets are used to estimate the efficiency
(relative to the number of problem functions evaluations)
of an algorithm to solve a problem (or several)
and computes its data profile (see :mod:`data_profile`).
"""
from typing import List, Optional

import matplotlib.pyplot as plt
from numpy import inf, linspace

from data_profiles.performance_history import PerformanceHistory


class TargetValues(PerformanceHistory):
    """Target values of a problem"""

    def compute_target_hits_history(
            self,
            values_history  # type: PerformanceHistory
    ):  # type: (...) -> List[int]
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

    def plot(
            self,
            show=True,  # type: bool
            path=None,  # type: Optional[str]
    ):  # type: (...) -> None
        """Compute and plot the target values.

        Args:
            show: If True, show the plot.
            path: The path where to save the plot.
                If None, the plot is not saved.
        """
        objective_values = [
            inf if item.infeasibility_measure > 0.0 else item.objective_value
            for item in self
        ]
        targets_number = len(self)
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.set_title("Target values")
        plt.xlabel("Target index")
        plt.xlim([0, targets_number + 1])
        plt.xticks(linspace(1, targets_number, dtype=int))
        plt.ylabel("Target value")
        axes.semilogy(
            range(1, targets_number + 1), objective_values, marker="o", linestyle=""
        )

        # Save and/or show the plot
        if path is not None:
            plt.savefig(path)
        if show:
            plt.show()
        plt.close()
