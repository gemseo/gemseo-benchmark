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

from __future__ import annotations


class HistoryItem:
    """A performance history item."""

    def __init__(
        self,
        performance_measure: float,
        infeasibility_measure: float,
        n_unsatisfied_constraints: int | None = None,
    ) -> None:
        """
        Args:
            performance_measure: The performance measure of the item.
            infeasibility_measure: The infeasibility measure of the item.
            n_unsatisfied_constraints: The number of unsatisfied constraints of the
                item.
                If ``None``, it will be set to 0 if the infeasibility measure is zero,
                and if the infeasibility measure is positive it will be set to None.
        """  # noqa: D205, D212, D415
        self.__performance_measure = performance_measure
        (
            self.__infeas_measure,
            self.__n_unsatisfied_constraints,
        ) = HistoryItem.__get_infeasibility(
            infeasibility_measure, n_unsatisfied_constraints
        )

    @staticmethod
    def __get_infeasibility(
        infeasibility_measure: float, n_unsatisfied_constraints: int | None
    ) -> tuple[float, int | None]:
        """Check the infeasibility measure and the number of unsatisfied constraints.

        Args:
            infeasibility_measure: The infeasibility measure.
            n_unsatisfied_constraints: The number of unsatisfied constraints.

        Returns:
            The infeasibility measure and the number of unsatisfied constraints.

        Raises:
             ValueError: If the infeasibility measure is negative,
                or if the number of unsatisfied constraints is negative,
                or if the infeasibility measure and the number of unsatisfied
                constraints are inconsistent.
        """
        if infeasibility_measure < 0.0:
            msg = f"The infeasibility measure is negative: {infeasibility_measure}."
            raise ValueError(msg)

        if n_unsatisfied_constraints is None:
            if infeasibility_measure == 0.0:
                return infeasibility_measure, 0
            return infeasibility_measure, None

        if n_unsatisfied_constraints < 0:
            msg = (
                "The number of unsatisfied constraints is negative: "
                f"{n_unsatisfied_constraints}."
            )
            raise ValueError(msg)

        if (infeasibility_measure == 0.0 and n_unsatisfied_constraints != 0) or (
            infeasibility_measure > 0.0 and n_unsatisfied_constraints == 0
        ):
            msg = (
                f"The infeasibility measure ({infeasibility_measure}) and the number "
                f"of unsatisfied constraints ({n_unsatisfied_constraints}) are not "
                f"consistent."
            )
            raise ValueError(msg)

        return infeasibility_measure, n_unsatisfied_constraints

    @property
    def performance_measure(self) -> float:
        """The performance measure of the history item."""
        return self.__performance_measure

    @property
    def infeasibility_measure(self) -> float:
        """The infeasibility measure of the history item.

        Raises:
             ValueError: If the infeasibility measure is negative.
        """
        return self.__infeas_measure

    @property
    def n_unsatisfied_constraints(self) -> int | None:
        """The number of unsatisfied constraints."""
        return self.__n_unsatisfied_constraints

    def __repr__(self) -> str:
        return str((self.performance_measure, self.infeasibility_measure))

    def __eq__(self, other: HistoryItem) -> bool:
        """Compare the history item with another one for equality.

        Args:
            other: The other history item.

        Returns:
            Whether the history item is equal to the other one.
        """
        return (
            self.__infeas_measure == other.__infeas_measure
            and self.performance_measure == other.performance_measure
        )

    def __lt__(self, other: HistoryItem) -> bool:
        """Compare the history item to another one for lower inequality.

        Args:
            other: The other history item.

        Returns:
            Whether the history item is lower than the other one.
        """
        return self.__infeas_measure < other.__infeas_measure or (
            self.__infeas_measure == other.__infeas_measure
            and self.performance_measure < other.performance_measure
        )

    def __le__(self, other: HistoryItem) -> bool:
        """Compare the history item to another one for lower inequality or equality.

        Args:
            other: The other history item.

        Returns:
            Whether the history item is lower than or equal to the other one.
        """
        return self < other or self == other

    @property
    def is_feasible(self) -> bool:
        """Whether the history item is feasible."""
        return self.infeasibility_measure == 0.0

    def apply_infeasibility_tolerance(self, infeasibility_tolerance: float) -> None:
        """Apply a tolerance on the infeasibility measure.

        Mark the history item as feasible if its infeasibility measure is below the
        tolerance.

        Args:
            infeasibility_tolerance: the tolerance on the infeasibility measure.
        """
        if self.__infeas_measure <= infeasibility_tolerance:
            self.__infeas_measure = 0.0
            self.__n_unsatisfied_constraints = 0

    def copy(self) -> HistoryItem:
        """Return a deep copy of the history item."""
        return HistoryItem(
            self.__performance_measure,
            self.__infeas_measure,
            self.__n_unsatisfied_constraints,
        )

    def switch_performance_measure_sign(self) -> None:
        """Switch the sign of the performance measure."""
        self.__performance_measure = -self.__performance_measure
