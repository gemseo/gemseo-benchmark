from functools import reduce
from itertools import chain, repeat
from math import ceil
from operator import itemgetter
from typing import Iterable, Iterator, List, Optional, Tuple

from numpy import inf


class PerformanceHistory(object):
    """Compare the values generated by an algorithm.

    Attributes:
        _values (List[float]): The history of performance measures.

    """

    def __init__(
            self,
            objective_values,  # type: List[float]
            infeasibility_measures=None,  # type: Optional[List[float]]
            feasibility=None  # type: Optional[List[bool]]
    ):
        """
        - If infeasibility measures are passed then the (optional) feasibility
          statuses are disregarded.
        - If no infeasibility measures are passed but feasibility statuses are passed
          then the infeasibility measures are set to zero in case of feasibility,
          and set to infinity otherwise.
        - If neither infeasibility measures nor feasibility statuses are passed then
          every infeasibility measure is set to zero.
        """
        if infeasibility_measures is not None:
            if len(infeasibility_measures) != len(objective_values):
                raise ValueError("Objective history and infeasibility measures history "
                                 "must have same length.")
            nonnegativity = all([a_val >= 0.0 for a_val in infeasibility_measures])
            if not nonnegativity:
                raise ValueError("Infeasibility measures must be non-negative.")
        elif feasibility is not None:
            if len(feasibility) != len(objective_values):
                raise ValueError("Objective history and feasibility history must have "
                                 "same length.")
            infeasibility_measures = [0.0 if a_feas else inf for a_feas in feasibility]
        else:
            infeasibility_measures = [0.0] * len(objective_values)
        self._values = list(zip(objective_values, infeasibility_measures))

    def __len__(self):  # type: (...) -> int
        """Return the length of the history"""
        return len(self._values)

    def __iter__(self):  # type: (...) -> Iterator[Tuple[float, float]]
        return iter(self._values)

    def __getitem__(self, item):  # type: (...) -> Tuple[float, float]
        return self._values[item]

    def __str__(self):  # type: (...) -> str
        return str(self.to_list())

    @staticmethod
    def _less_equal(
            an_item,  # type:Tuple[float, float]
            another_item  # type:Tuple[float,float]
    ):  # type: (...) -> bool
        """Return whether a history item is lower or equal to another history item.

        Args:
            an_item: a history item
            another_item: another history item

        Returns:
            Whether the first history item is lower or equal to the second one.

        """
        a_value, a_meas = an_item
        other_value, other_meas = another_item
        if a_meas < 0 or other_meas < 0:
            raise ValueError("Negative infeasibility measure")
        return a_meas < other_meas or (a_meas == other_meas and a_value <= other_value)

    @staticmethod
    def _min(
            an_item,  # type: Tuple[float, float]
            another_item  # type: Tuple[float, float]
    ):  # type: (...)-> Tuple[float, float]
        """Return the smallest of two history items.

        Args:
            an_item: A history item.
            another_item: Another history item.

        Returns:
            The smallest of the two history items.

        """
        if PerformanceHistory._less_equal(an_item, another_item):
            return an_item
        else:
            return another_item

    def cumulated_min_history(
            self,
            fill_up_to=None  # type: Optional[int]
    ):  # type: (...) -> PerformanceHistory
        """Return the history of the cumulated minimum.

        Args:
            fill_up_to: Optionally, the last value of the minima history will be append
                as many times as necessary for the minima history to be of the required
                length.

        Returns:
            The history of the cumulated minimum.

        """
        minima = [reduce(PerformanceHistory._min, self._values[:i + 1])
                  for i in range(len(self))]
        if fill_up_to is not None and fill_up_to < len(minima):
            raise ValueError("Cannot fill up to length {} lower than total length {}"
                             .format(fill_up_to, len(minima)))
        elif fill_up_to is not None:
            minima = list(chain(minima, repeat(minima[-1], (fill_up_to - len(minima)))))
        return PerformanceHistory(*zip(*minima))

    def to_list(self):  # type: (...) -> List[Tuple[float, float]]
        """Return the performance history as a list of 2-tuples.

        Returns:
            The performance history as a list.

        """
        return self._values

    def sorted(self):  # type: (...) -> PerformanceHistory
        """Return the sorted history of performance values"""
        return sorted(self._values, key=itemgetter(1, 0), reverse=True)

    def _median(self):  # type: (...) -> Tuple[float, float]
        """Return the median of the history of performance values"""
        return self.sorted()[ceil(len(self) // 2)]

    @staticmethod
    def median_history(
            histories  # type: Iterable[PerformanceHistory]
    ):  # type: (...) -> PerformanceHistory
        """Return the history of the median of several performance histories.

        Args:
            histories: The performance histories

        Returns:
            The median history.

        """
        histories_as_list = [a_hist.to_list() for a_hist in histories]
        median_as_list = list()
        for snapshot in zip(*histories_as_list):
            values_history, measures_history = zip(*snapshot)
            median = PerformanceHistory(values_history, measures_history)._median()
            median_as_list.append(median)
        median_history = PerformanceHistory(*zip(*median_as_list))
        return median_history
