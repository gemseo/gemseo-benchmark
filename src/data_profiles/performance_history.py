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
"""Class that generates performance measures out of data generated by an algorithm.

Iterative algorithms that solve, for example, optimization problems or equations
produce histories of data such as the value of the objective to minimize,
or the size of the equation residual, at each iteration.
The best value obtained up until each iteration can be generated out of this data.
Here we call "performance history" the history of the best values obtained up until
each iteration.

Infeasible data can be discarded based upon histories of infeasibility measures or
boolean feasibility statuses.

Performance histories can be used to generate target values for a problem,
or to generate the data profile of an algorithm.
"""
import json
from functools import reduce
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from numpy import inf

from data_profiles.history_item import HistoryItem


class PerformanceHistory(Sequence[HistoryItem]):
    """A history of performance measures generated by an algorithm.

    A `PerformanceHistory` is a sequence of `HistoryItem`s.

    Attributes:
        history_items (List[HistoryItem]): The items of the performance history.
    """
    PERFORMANCE = "performance"
    INFEASIBILITY = "infeasibility"

    def __init__(
            self,
            objective_values=None,  # type: Optional[Sequence[float]]
            infeasibility_measures=None,  # type: Optional[Sequence[float]]
            feasibility_statuses=None  # type: Optional[Sequence[bool]]
    ):  # type: (...) -> None
        """
        Args:
            objective_values: The history of the quantity to be minimized.
            infeasibility_measures: The history of infeasibility measures.
                An infeasibility measure is a non-negative real number representing
                the gap between the design and the feasible space,
                a zero value meaning feasibility.
                If None and `feasibility_history` is not None
                then the infeasibility measures are set to zero in case of feasibility,
                and set to infinity otherwise.
                If None and `feasibility_history` is None
                then every infeasibility measure is set to zero.
            feasibility_statuses: The history of the (boolean) feasibility statuses.
                If `infeasibility_measures` is not None then `feasibility_statuses` is
                disregarded.

        Raises:
            ValueError: If the lengths of the histories do not match.
        """
        if objective_values is None:
            objective_values = []
        if infeasibility_measures is not None:
            if len(infeasibility_measures) != len(objective_values):
                raise ValueError("The objective history and the infeasibility history "
                                 "must have same length.")
        elif feasibility_statuses is not None:
            if len(feasibility_statuses) != len(objective_values):
                raise ValueError("The objective history and the feasibility history "
                                 "must have same length.")
            infeasibility_measures = [
                0.0 if is_feas else inf for is_feas in feasibility_statuses
            ]
        else:
            infeasibility_measures = [0.0] * len(objective_values)
        self.history_items = [
            HistoryItem(value, measure) for value, measure
            in zip(objective_values, infeasibility_measures)
        ]

    @property
    def objective_values(self):  # type: (...) -> List[float]
        """The objective values."""
        return [item.objective_value for item in self.history_items]

    @property
    def infeasibility_measures(self):  # type: (...) -> List[float]
        """The infeasibility measures."""
        return [item.infeasibility_measure for item in self.history_items]

    @property
    def history_items(self):  # type: (...) -> List[HistoryItem]
        """The history items.

        Raises:
            TypeError: If an item is set with a type different from HistoryItem.
        """
        return self.__items

    @history_items.setter
    def history_items(
            self,
            history_items,  # type: Iterable[HistoryItem]
    ):  # type: (...) -> None
        for item in history_items:
            if not isinstance(item, HistoryItem):
                raise TypeError("History items must be of type HistoryItem")
        self.__items = list(history_items)

    def __len__(self):  # type: (...) -> int
        return len(self.__items)

    def __getitem__(
            self,
            i,  # type: int
    ):  # type: (...) -> HistoryItem
        return self.__items[i]

    def __repr__(self):  # type: (...) -> str
        return str([item for item in self])

    def compute_cumulated_minimum(self):  # type: (...) -> PerformanceHistory
        """Return the history of the cumulated minimum.

        Returns:
            The history of the cumulated minimum.
        """
        minima = [reduce(min, self.__items[:i + 1]) for i in range(len(self))]
        minimum_history = PerformanceHistory()
        minimum_history.history_items = minima
        return minimum_history

    def __compute_median(self):  # type: (...) -> HistoryItem
        """Return the median of the history of performance values.

        Returns:
            The median of the history of performance values.
        """
        # Compute the middle index of the items (N.B. zero-based index)
        if len(self) % 2 == 0:
            middle_index = len(self) // 2 - 1
        else:
            middle_index = len(self) // 2
        return sorted(self.__items)[middle_index]

    @staticmethod
    def compute_median_history(
            histories  # type: Iterable[PerformanceHistory]
    ):  # type: (...) -> PerformanceHistory
        """Return the history of the median of several performance histories.

        Args:
            histories: The performance histories

        Returns:
            The median history.
        """
        medians = list()
        # Iterate over the sets of history items corresponding to the same iteration
        # and compute their medians
        for snapshot in zip(*[hist.history_items for hist in histories]):
            snapshot_as_hist = PerformanceHistory()
            snapshot_as_hist.history_items = snapshot
            median = snapshot_as_hist.__compute_median()
            medians.append(median)
        median_history = PerformanceHistory()
        median_history.history_items = medians
        return median_history

    def remove_leading_infeasible(self):  # type: (...) -> PerformanceHistory
        """Return the history starting from the first feasible item.

        Returns:
            The truncated performance history.
        """
        first_feasible = None
        for index, item in enumerate(self):
            if item.infeasibility_measure == 0.0:
                first_feasible = index
                break
        truncated_history = PerformanceHistory()
        if first_feasible is not None:
            truncated_history.history_items = self.history_items[first_feasible:]
        return truncated_history

    def save_to_file(
            self,
            file_path,  # type: Union[str, Path]
    ):  # type: (...) -> None
        """Save the performance history in a file.

        Args:
            file_path: The path where to write the file.
        """
        data = list()
        # Add each history item in dictionary format
        for item in self.history_items:
            data_item = {
                PerformanceHistory.PERFORMANCE: item.objective_value,
                PerformanceHistory.INFEASIBILITY: item.infeasibility_measure
            }
            data.append(data_item)
        with Path(file_path).open("w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def from_file(
            cls,
            file_path,  # type: Union[str, Path]
    ):  # type: (...) -> PerformanceHistory
        """Create a new performance history from a file.

        Args:
            file_path: The path to the file.

        Returns:
            The performance history.
        """
        with Path(file_path).open("r") as file:
            data = json.load(file)
        objective_values = list()
        infeasibility_measures = list()
        for item in data:
            objective_values.append(item[PerformanceHistory.PERFORMANCE])
            infeasibility_measures.append(item[PerformanceHistory.INFEASIBILITY])
        return cls(objective_values, infeasibility_measures)
