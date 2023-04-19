# Copyright 2022 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
from __future__ import annotations

import json
import statistics
from functools import reduce
from itertools import chain
from itertools import repeat
from pathlib import Path
from typing import Callable
from typing import Iterable
from typing import Sequence

from gemseo.algos.opt_problem import OptimizationProblem
from matplotlib.axes import Axes
from numpy import inf

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.results.history_item import HistoryItem


class PerformanceHistory(Sequence[HistoryItem]):
    """A history of performance measures generated by an algorithm.

    A `PerformanceHistory` is a sequence of `HistoryItem`s.

    Attributes:
        problem_name (str): The name of the problem.
        total_time (float): The run time of the algorithm.
    """

    __ALGORITHM_CONFIGURATION = "algorithm_configuration"
    __DOE_SIZE = "DOE_size"
    __EXECUTION_TIME = "execution_time"
    __HISTORY_ITEMS = "history_items"
    __INFEASIBILITY = "infeasibility"
    __N_UNSATISFIED_CONSTRAINTS = "n_unsatisfied_constraints"
    __PERFORMANCE = "performance"
    __PROBLEM = "problem"

    def __init__(
        self,
        objective_values: Sequence[float] = None,
        infeasibility_measures: Sequence[float] = None,
        feasibility_statuses: Sequence[bool] = None,
        n_unsatisfied_constraints: Sequence[int] = None,
        problem_name: str = None,
        objective_name: str = None,
        constraints_names: Sequence[str] = None,
        doe_size: int = None,
        total_time: float = None,
        algorithm_configuration: AlgorithmConfiguration = None,
    ) -> None:
        """
        Args:
            objective_values: The history of the quantity to be minimized.
                If ``None``, will be considered empty.
            infeasibility_measures: The history of infeasibility measures.
                An infeasibility measure is a non-negative real number representing
                the gap between the design and the feasible space,
                a zero value meaning feasibility.
                If ``None`` and `feasibility_statuses` is not None
                then the infeasibility measures are set to zero in case of feasibility,
                and set to infinity otherwise.
                If ``None`` and `feasibility_statuses` is None
                then every infeasibility measure is set to zero.
            feasibility_statuses: The history of the (boolean) feasibility statuses.
                If `infeasibility_measures` is not None then `feasibility_statuses` is
                disregarded.
                If ``None`` and 'infeasibility_measures' is None
                then every infeasibility measure is set to zero.
            n_unsatisfied_constraints: The history of the number of unsatisfied
                constraints.
                If ``None``, the entries will be set to 0 for feasible entries
                and None for infeasible entries.
            problem_name: The name of the problem.
                If ``None``, it will not be set.
            objective_name: The name of the objective function.
                If ``None``, it will not be set.
            constraints_names: The names the scalar constraints.
                Each name must correspond to a scalar value.
                If ``None``, it will not be set.
            doe_size: The size of the initial design of experiments.
                If ``None``, it will not be set.
            total_time: The total time of the optimization, in seconds.
                If ``None``, it will not be set.
            algorithm_configuration: The name of the algorithm which generated the
                history.
                If ``None``, it will not be set.

        Raises:
            ValueError: If the lengths of the histories do not match.
        """  # noqa: D205, D212, D415
        if constraints_names is None:
            self._constraints_names = []
        else:
            self._constraints_names = constraints_names

        self._objective_name = objective_name
        self.algorithm_configuration = algorithm_configuration
        self.doe_size = doe_size
        self.items = self.__get_history_items(
            objective_values,
            infeasibility_measures,
            feasibility_statuses,
            n_unsatisfied_constraints,
        )
        self.problem_name = problem_name
        self.total_time = total_time

    @property
    def objective_values(self) -> list[float]:
        """The objective values."""
        return [item.objective_value for item in self.items]

    @property
    def infeasibility_measures(self) -> list[float]:
        """The infeasibility measures."""
        return [item.infeasibility_measure for item in self.items]

    @property
    def n_unsatisfied_constraints(self) -> list[int]:
        """The numbers of unsatisfied constraints."""
        return [item.n_unsatisfied_constraints for item in self.items]

    @property
    def items(self) -> list[HistoryItem]:
        """The history items.

        Raises:
            TypeError: If an item is set with a type different from HistoryItem.
        """
        return self.__items

    @items.setter
    def items(
        self,
        history_items: Iterable[HistoryItem],
    ) -> None:
        for item in history_items:
            if not isinstance(item, HistoryItem):
                raise TypeError(
                    "History items must be of type HistoryItem."
                    f" The following type was passed: {type(item)}."
                )

        self.__items = list(history_items)

    @staticmethod
    def __get_history_items(
        objective_values: Sequence[float] = None,
        infeasibility_measures: Sequence[float] = None,
        feasibility_statuses: Sequence[bool] = None,
        n_unsatisfied_constraints: Sequence[int] = None,
    ) -> list[HistoryItem]:
        """Return history items based on values histories.

        Args:
            objective_values: The history of the quantity to be minimized.
                If ``None``, will be considered empty.
            infeasibility_measures: The history of infeasibility measures.
                An infeasibility measure is a non-negative real number representing
                the gap between the design and the feasible space,
                a zero value meaning feasibility.
                If ``None`` and `feasibility_statuses` is not None
                then the infeasibility measures are set to zero in case of feasibility,
                and set to infinity otherwise.
                If ``None`` and `feasibility_statuses` is None
                then every infeasibility measure is set to zero.
            feasibility_statuses: The history of the (boolean) feasibility statuses.
                If `infeasibility_measures` is not None then `feasibility_statuses` is
                disregarded.
                If ``None`` and 'infeasibility_measures' is None
                then every infeasibility measure is set to zero.
            n_unsatisfied_constraints: The history of the number of unsatisfied
                constraints.
                If ``None``, the entries will be set to 0 for feasible entries
                and None for infeasible entries.

        Returns:
            The history items.
        """
        if objective_values is None:
            objective_values = []

        if infeasibility_measures is not None:
            if len(infeasibility_measures) != len(objective_values):
                raise ValueError(
                    "The objective history and the infeasibility history "
                    "must have same length."
                )
        elif feasibility_statuses is not None:
            if len(feasibility_statuses) != len(objective_values):
                raise ValueError(
                    "The objective history and the feasibility history "
                    "must have same length."
                )
            infeasibility_measures = [
                0.0 if is_feas else inf for is_feas in feasibility_statuses
            ]
        else:
            infeasibility_measures = [0.0] * len(objective_values)

        if n_unsatisfied_constraints is None:
            n_unsatisfied_constraints = [
                0 if entry == 0.0 else None for entry in infeasibility_measures
            ]
        elif len(n_unsatisfied_constraints) != len(infeasibility_measures):
            raise ValueError(
                "The unsatisfied constraints history and the feasibility history"
                " must have same length."
            )

        return [
            HistoryItem(value, measure, n_unsatisfied)
            for value, measure, n_unsatisfied in zip(
                objective_values, infeasibility_measures, n_unsatisfied_constraints
            )
        ]

    def __len__(self) -> int:
        return len(self.__items)

    def __getitem__(
        self,
        i: int,
    ) -> HistoryItem:
        return self.__items[i]

    def __repr__(self) -> str:
        return str([item for item in self])

    def compute_cumulated_minimum(self) -> PerformanceHistory:
        """Return the history of the cumulated minimum.

        Returns:
            The history of the cumulated minimum.
        """
        minima = [reduce(min, self.__items[: i + 1]) for i in range(len(self))]
        minimum_history = PerformanceHistory()
        minimum_history.items = minima
        return minimum_history

    @staticmethod
    def compute_minimum_history(
        histories: Iterable[PerformanceHistory],
    ) -> PerformanceHistory:
        """Return the minimum of several performance histories.

        Args:
            histories: The performance histories

        Returns:
            The minimum history.
        """
        return PerformanceHistory.__compute_statistic(histories, min)

    @staticmethod
    def compute_maximum_history(
        histories: Iterable[PerformanceHistory],
    ) -> PerformanceHistory:
        """Return the maximum of several performance histories.

        Args:
            histories: The performance histories

        Returns:
            The maximum history.
        """
        return PerformanceHistory.__compute_statistic(histories, max)

    @staticmethod
    def compute_median_history(
        histories: Iterable[PerformanceHistory],
    ) -> PerformanceHistory:
        """Return the median of several performance histories.

        Args:
            histories: The performance histories

        Returns:
            The median history.
        """
        return PerformanceHistory.__compute_statistic(histories, statistics.median_low)

    @staticmethod
    def __compute_statistic(
        histories: Iterable[PerformanceHistory],
        statistic: Callable[[tuple[HistoryItem]], HistoryItem],
    ) -> PerformanceHistory:
        """Return the history of a statistic of several performance histories.

        The histories are extended to the same length before being split.

        Args:
            histories: The performance histories.
            statistic: The statistic.

        Returns:
            The tuples of HistoryItems.
        """
        # Extend the histories to the same length
        budget_max = max(len(history) for history in histories)
        extended_histories = [history.extend(budget_max) for history in histories]

        # Create the performance history
        history = PerformanceHistory()
        history.items = [
            statistic(items)
            for items in zip(*[history.items for history in extended_histories])
        ]
        return history

    def remove_leading_infeasible(self) -> PerformanceHistory:
        """Return the history starting from the first feasible item.

        Returns:
            The truncated performance history.
        """
        first_feasible = None
        for index, item in enumerate(self):
            if item.is_feasible:
                first_feasible = index
                break

        truncated_history = PerformanceHistory()
        if first_feasible is not None:
            truncated_history.items = self.items[first_feasible:]

        return truncated_history

    def to_file(
        self,
        path: str | Path,
    ) -> None:
        """Save the performance history in a file.

        Args:
            path: The path where to write the file.
        """
        items_data = list()
        # Add each history item in dictionary format
        for item in self.items:
            data_item = {
                PerformanceHistory.__PERFORMANCE: item.objective_value,
                PerformanceHistory.__INFEASIBILITY: item.infeasibility_measure,
            }
            if item.n_unsatisfied_constraints is not None:
                # N.B. type int64 is not JSON serializable
                data_item[PerformanceHistory.__N_UNSATISFIED_CONSTRAINTS] = int(
                    item.n_unsatisfied_constraints
                )

            items_data.append(data_item)

        data = dict()
        if self.problem_name is not None:
            data[self.__PROBLEM] = self.problem_name

        if self.algorithm_configuration is not None:
            data[
                self.__ALGORITHM_CONFIGURATION
            ] = self.algorithm_configuration.to_dict()

        if self.doe_size is not None:
            data[self.__DOE_SIZE] = self.doe_size

        if self.total_time is not None:
            data[self.__EXECUTION_TIME] = self.total_time

        data[self.__HISTORY_ITEMS] = items_data
        with Path(path).open("w") as file:
            json.dump(data, file, indent=4, separators=(",", ": "))

    @classmethod
    def from_file(cls, path: str | Path) -> PerformanceHistory:
        """Create a new performance history from a file.

        Args:
            path: The path to the file.

        Returns:
            The performance history.
        """
        with Path(path).open("r") as file:
            data = json.load(file)

        # Cover deprecated performance history files
        if isinstance(data, list):
            history = cls()
            history.items = [
                HistoryItem(
                    item_data[PerformanceHistory.__PERFORMANCE],
                    item_data[PerformanceHistory.__INFEASIBILITY],
                    item_data.get(PerformanceHistory.__N_UNSATISFIED_CONSTRAINTS),
                )
                for item_data in data
            ]
            return history

        history = cls()
        history.problem_name = data.get(cls.__PROBLEM)
        if cls.__ALGORITHM_CONFIGURATION in data:
            history.algorithm_configuration = AlgorithmConfiguration.from_dict(
                data[cls.__ALGORITHM_CONFIGURATION]
            )

        history.doe_size = data.get(cls.__DOE_SIZE)
        history.total_time = data.get(cls.__EXECUTION_TIME)
        history.items = [
            HistoryItem(
                item_data[PerformanceHistory.__PERFORMANCE],
                item_data[PerformanceHistory.__INFEASIBILITY],
                item_data.get(PerformanceHistory.__N_UNSATISFIED_CONSTRAINTS),
            )
            for item_data in data[cls.__HISTORY_ITEMS]
        ]
        return history

    @classmethod
    def from_problem(
        cls,
        problem: OptimizationProblem,
        problem_name: str = None,
    ) -> PerformanceHistory:
        """Create a performance history from a solved optimization problem.

        Args:
            problem: The optimization problem.
            problem_name: The name of the problem.
                If ``None``, the name of the problem is not set.

        Returns:
            The performance history.
        """
        obj_name = problem.objective.name
        obj_values = list()
        infeas_measures = list()
        feas_statuses = list()
        n_unsatisfied_constraints = list()
        functions_names = set([obj_name] + problem.get_constraint_names())
        for design_values, output_values in problem.database.items():
            # Only consider points with all functions values
            if not functions_names <= set(output_values.keys()):
                continue

            x_vect = design_values.unwrap()
            obj_values.append(float(output_values[obj_name]))
            feasibility, measure = problem.get_violation_criteria(x_vect)
            number_of_unsatisfied_constraints = (
                problem.get_number_of_unsatisfied_constraints(x_vect)
            )
            infeas_measures.append(measure)
            feas_statuses.append(feasibility)
            n_unsatisfied_constraints.append(number_of_unsatisfied_constraints)

        return cls(
            obj_values,
            infeas_measures,
            feas_statuses,
            n_unsatisfied_constraints,
            problem_name,
            problem.objective.name,
            problem.get_scalar_constraint_names(),
        )

    def get_plot_data(
        self, feasible: bool = False, minimum_history: bool = False
    ) -> tuple[list[int], list[HistoryItem]]:
        """Return the data to plot the performance history.

        Args:
            feasible: Whether to get only feasible values.
            minimum_history: Whether to get the history of the cumulated minimum
                instead of the history of the objective value.

        Returns:
            The abscissas and the ordinates of the plot.
        """
        if minimum_history:
            history = self.compute_cumulated_minimum()
        else:
            history = self

        # Find the index of the first feasible history item
        if feasible:
            first_feasible_index = len(history)
            for index, item in enumerate(history):
                if item.is_feasible:
                    first_feasible_index = index
                    break
        else:
            first_feasible_index = 0

        return (
            list(range(first_feasible_index + 1, len(history) + 1)),
            history[first_feasible_index:],
        )

    def extend(self, size: int) -> PerformanceHistory:
        """Extend the performance history by repeating its last item.

        If the history is longer than the expected size then it will not be altered.

        Args:
            size: The expected size of the extended performance history.

        Returns:
            The extended performance history.

        Raises:
            ValueError: If the expected size is smaller than the history size.
        """
        if size < len(self):
            raise ValueError(
                f"The expected size ({size}) is smaller than "
                f"the history size ({len(self)})."
            )

        history = PerformanceHistory()
        history.items = list(chain(self, repeat(self[-1], (size - len(self)))))
        return history

    def shorten(self, size: int) -> PerformanceHistory:
        """Shorten the performance history to a given size.

        If the history is shorter than the expected size then it will not be altered.

        Args:
            size: The expected size of the shortened performance history.

        Returns:
            The shortened performance history.
        """
        history = PerformanceHistory()
        history.items = self.items[:size]
        return history

    def plot(self, axes: Axes, only_feasible: bool, **kwargs: str | float) -> None:
        """Plot the performance history.

        Args:
            axes: The axes on which to plot the performance history.
            only_feasible: Whether to plot the feasible items only.
            **kwargs: The options to be passed to Axes.plot.
        """
        abscissas, history_items = self.get_plot_data(feasible=only_feasible)
        ordinates = [item.objective_value for item in history_items]
        axes.plot(abscissas, ordinates, **kwargs)

    def apply_infeasibility_tolerance(self, infeasibility_tolerance: float) -> None:
        """Apply a tolerance on the infeasibility measures of the history items.

        Mark the history items with an infeasibility measure below the tolerance
        as feasible.

        Args:
            infeasibility_tolerance: the tolerance on the infeasibility measure.
        """
        for item in self.items:
            item.apply_infeasibility_tolerance(infeasibility_tolerance)
