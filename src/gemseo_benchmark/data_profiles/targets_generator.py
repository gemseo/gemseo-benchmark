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
"""Generation of targets for a problem to be solved by an iterative algorithm.

The targets are generated out of algorithms histories considered to be of reference: the
median of the reference histories is computed and a uniformly distributed subset (of the
required size) of this median history is extracted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

import matplotlib.pyplot as plt
from gemseo.utils.matplotlib_figure import save_show_figure
from matplotlib.ticker import MaxNLocator
from numpy import linspace
from numpy import ndarray

from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.results.history_item import HistoryItem
from gemseo_benchmark.results.performance_histories import PerformanceHistories
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.figure import Figure


class TargetsGenerator:
    """Compute the target values for an objective to minimize."""

    __NO_HISTORIES_MESSAGE: Final[str] = (
        "There are no histories to generate the targets from."
    )

    __histories: PerformanceHistories
    """A collection of performance histories."""

    def __init__(self) -> None:  # noqa: D107
        self.__histories = PerformanceHistories()

    def add_history(
        self,
        objective_values: Sequence[float] | None = None,
        infeasibility_measures: Sequence[float] | None = None,
        feasibility_statuses: Sequence[bool] | None = None,
        history: PerformanceHistory | None = None,
    ) -> None:
        """Add a history of objective values.

        Args:
            objective_values: A history of objective values.
                If ``None``, a performance history must be passed.
                N.B. the value at index i is assumed to have been obtained with i+1
                evaluations.
            infeasibility_measures: A history of infeasibility measures.
                If ``None`` then measures are set to zero in case of feasibility and set
                to infinity otherwise.
            feasibility_statuses: A history of (boolean) feasibility statuses.
                If ``None`` then feasibility is always assumed.
            history: A performance history.
                If ``None``, objective values must be passed.

        Raises:
            ValueError: If neither a performance history nor objective values are
                passed, or if both are passed.
        """
        if history is not None:
            if objective_values is not None:
                raise ValueError(
                    "Both a performance history and objective values were passed."
                )
        elif objective_values is None:
            raise ValueError(
                "Either a performance history or objective values must be passed."
            )
        else:
            history = PerformanceHistory(
                objective_values, infeasibility_measures, feasibility_statuses
            )
        self.__histories.append(history)

    def compute_target_values(
        self,
        targets_number: int,
        budget_min: int = 1,
        feasible: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        best_target_objective: float | None = None,
        best_target_tolerance: float = 0.0,
    ) -> TargetValues:
        """Compute the target values for a function from the histories of its values.

        Args:
            targets_number: The number of targets to compute.
            budget_min: The number of functions evaluations to be used to define the
                first target.
                If argument ``feasible`` is set to ``True``, this argument will be
                disregarded and the evaluation budget defining the easiest target
                will be the budget of the first item in the histories reaching the
                best target value.
            feasible: Whether to generate only feasible targets.
            show: Whether to show the plot.
            file_path: The file path to save the plot.
                If ``None``, the plot is not saved.
            best_target_objective: The objective value of the best target value.
                If ``None``, it will be inferred from the performance histories.
            best_target_tolerance: The relative tolerance for comparison with the
                best target value.

        Returns:
            The target values of the function.

        Raises:
            RuntimeError: If feasibility is required but the best target value is not
                feasible.
        """
        # Get the performance histories of reference
        reference_histories, best_target = self.__get_reference_histories(
            self.__histories, best_target_objective, best_target_tolerance, feasible
        )

        # Compute the median of the cumulated minimum histories
        median_history = PerformanceHistories(*reference_histories).compute_median()
        if feasible:
            median_history = median_history.remove_leading_infeasible()

        # Truncate the values that stagnate near the best target
        for index, item in enumerate(median_history):
            if item <= best_target:
                median_history = median_history[: index + 1]
                break

        # Compute a budget scale
        budget_scale = self.__compute_budget_scale(
            budget_min, len(median_history), targets_number
        )

        # Compute the target values
        target_values = TargetValues()
        target_values.items = [median_history[item - 1] for item in budget_scale]

        # Plot the target values
        if show or file_path is not None:
            target_values.plot(show, file_path)

        return target_values

    @staticmethod
    def __compute_budget_scale(
        budget_min: int,
        budget_max: int,
        budgets_number: int,
    ) -> ndarray:
        """Compute a scale of evaluation budgets.

         The progression of the scale relates to complexity in terms of evaluation cost.

        N.B. here the evaluation cost is assumed linear with respect to the number of
        evaluations.

        Args:
            budget_min: The minimum number of evaluations.
            budget_max: The maximum number of evaluations.
            budgets_number: The number of budgets.

        Returns:
            The distribution of evaluation budgets.

        Raises:
            ValueError: If the number of targets required is larger
                than the size the longest history
                starting from budget_min.
        """
        if budgets_number > budget_max - budget_min + 1:
            raise ValueError(
                f"The number of targets required ({budgets_number}) is greater "
                f"than the size the longest history ({budget_max - budget_min + 1}) "
                f"starting from budget_min ({budget_min})."
            )

        return linspace(budget_min, budget_max, budgets_number, dtype=int)

    @staticmethod
    def __get_best_target(
        objective_value: float,
        infeasibility_measure: float,
        tolerance: float,
    ) -> HistoryItem:
        """Return the best target value.

        Args:
            objective_value: The objective value of the best target value.
            infeasibility_measure: The infeasibility measure of the best target value.
            tolerance: The tolerance for comparisons with the best target value.

        Returns:
            The best target value.
        """
        if infeasibility_measure == 0.0:
            return HistoryItem(
                objective_value + max(tolerance * abs(objective_value), tolerance),
                infeasibility_measure,
            )

        return HistoryItem(
            objective_value,
            infeasibility_measure + tolerance * abs(infeasibility_measure),
        )

    @staticmethod
    def __get_reference_histories(
        histories: PerformanceHistories,
        best_target_objective: float | None,
        best_target_tolerance: float,
        feasible: bool,
    ) -> tuple[PerformanceHistories, HistoryItem]:
        """Return the performance histories of reference.

        1. Compute the histories of the cumulated minima.
        2. Select the histories that reach the best target.

        Args:
            histories: The performance histories.
            best_target_objective: The objective value of the best target.
            best_target_tolerance: The tolerance for comparison with the best target.
            feasible: Whether the best target must be feasible.

        Returns:
             The histories of the cumulated minima.

        Raises:
            RuntimeError: If there are no performance histories from which to compute
                the target values.
        """
        if not histories:
            raise RuntimeError(TargetsGenerator.__NO_HISTORIES_MESSAGE)

        # Get the histories of the cumulated minima
        reference_histories = histories.cumulate_minimum()

        # Get the best target value
        if best_target_objective is None:
            best_item = min([history[-1] for history in reference_histories])
            best_target = TargetsGenerator.__get_best_target(
                best_item.objective_value,
                best_item.infeasibility_measure,
                best_target_tolerance,
            )
        else:
            best_target = TargetsGenerator.__get_best_target(
                best_target_objective, 0.0, best_target_tolerance
            )

        if feasible and not best_target.is_feasible:
            raise RuntimeError("The best target value is not feasible.")

        # Get the performance histories that reach the best target value
        reference_histories = PerformanceHistories(*[
            history for history in reference_histories if history[-1] <= best_target
        ])
        if not reference_histories:
            raise RuntimeError(
                "There is no performance history that reaches the best target value."
            )

        return reference_histories, best_target

    def plot_histories(
        self,
        best_target_value: float | None = None,
        show: bool = False,
        file_path: str | Path | None = None,
    ) -> Figure:
        """Plot the histories used as a basis to compute the target values.

        Args:
            best_target_value: The best target value
                to be represented with a horizontal line.
                If ``None``, no best target value will be plotted.
            show: Whether to show the figure.
            file_path: The path where to save the figure.
                If ``None``, the figure will not be saved.

        Returns:
            The histories figure.
        """
        # Set up the figure
        figure = plt.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.set_title("Reference performance histories")
        plt.xlabel("Number or evaluations")
        plt.ylabel("Performance value")

        # Plot the best target value
        if best_target_value is not None:
            plt.axhline(y=best_target_value, color="r", linestyle="-")

        # Plot the histories of the cumulated minima
        maximum_budget = max(len(history) for history in self.__histories)
        minimum_budget = maximum_budget
        for history in self.__histories:
            budgets, items = history.get_plot_data(feasible=True, minimum_history=True)
            # Update the minimum budget
            if budgets:  # empty if there is no feasible points
                minimum_budget = min(budgets[0], minimum_budget)

            axes.plot(
                budgets,
                [item.objective_value for item in items],
                marker="o",
                linestyle=":",
            )

        plt.xlim(left=minimum_budget - 1, right=maximum_budget + 1)
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        save_show_figure(figure, show, file_path)
        return figure
