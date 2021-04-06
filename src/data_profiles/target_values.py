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
from typing import List

from data_profiles.performance_history import PerformanceHistory


class TargetValues(PerformanceHistory):
    """Target values of a problem"""

    def count_targets_hits(
            self,
            values_history  # type: PerformanceHistory
    ):  # type: (...) -> List[int]
        """Compute the history of the number of target hits associated with a history
        of values.

        Args:
            values_history: The history of values.

        Returns:
            The history of the number of target hits.

        """
        minimum_history = values_history.cumulated_min_history()
        hits_history = [[self._less_equal(a_minimum, a_target)
                         for a_target in self].count(True)
                        for a_minimum in minimum_history]
        return hits_history
