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
"""A performance history item."""
from typing import Optional


class HistoryItem(object):
    """A performance history item.

    Attributes:
        objective_value (float): The objective function value of the item.
        infeasibility_measure (float): The infeasibility measure of the item.
        n_unsatisfied_constraints (Optional[int]): The number of unsatisfied constraints
            of the item.
    """

    def __init__(
            self,
            objective_value,  # type: float
            infeasibility_measure,  # type: float
            n_unsatisfied_constraints=None,  # type: Optional[int]
    ):  # type: (...) -> None
        """
        Args:
            objective_value: The objective function value of the item.
            infeasibility_measure: The infeasibility measure of the item.
            n_unsatisfied_constraints: The number of unsatisfied constraints of the
                item.
        """
        self.objective_value = objective_value
        self.infeasibility_measure = infeasibility_measure
        self.n_unsatisfied_constraints = n_unsatisfied_constraints

    @property
    def infeasibility_measure(self):  # type: (...) -> float
        """The infeasibility measure of the history item.

        Raises:
             ValueError: If the infeasibility measure is negative.
        """
        return self.__infeas_measure

    @infeasibility_measure.setter
    def infeasibility_measure(
            self,
            infeasibility_measure,  # type: float
    ):  # type: (...) -> None
        if infeasibility_measure < 0.0:
            raise ValueError("The infeasibility measure must be non-negative.")
        self.__infeas_measure = infeasibility_measure

    def __repr__(self):  # type: (...) -> str
        return str((self.objective_value, self.infeasibility_measure))

    def __eq__(
            self,
            other,  # type: HistoryItem
    ):  # type: (...) -> bool
        """Compare the history item with another one for equality.

        Args:
            other: The other history item.

        Returns:
            Whether the history item is equal to the other one.
        """
        return (
                self.__infeas_measure == other.__infeas_measure
                and self.objective_value == other.objective_value
                and self.n_unsatisfied_constraints == other.n_unsatisfied_constraints
        )

    def __lt__(
            self,
            other,  # type: HistoryItem
    ):  # type: (...) -> bool
        """Compare the history item to another one for lower inequality.

        Args:
            other: The other history item.

        Returns:
            Whether the history item is lower than the other one.
        """
        return self.__infeas_measure < other.__infeas_measure or (
                self.__infeas_measure == other.__infeas_measure
                and self.objective_value < other.objective_value
        )

    def __le__(
            self,
            other,  # type: HistoryItem
    ):  # type: (...) -> bool
        """Compare the history item to another one for lower inequality or equality.

        Args:
            other: The other history item.

        Returns:
            Whether the history item is lower than or equal to the other one.
        """
        return self < other or self == other
