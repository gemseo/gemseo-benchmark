from typing import List, Optional

from matplotlib.pyplot import figure, semilogy, show, xlabel, xlim, xticks, ylabel
from numpy import array, inf, linspace, ndarray

from data_profiles.performance_history import PerformanceHistory
from data_profiles.target_values import TargetValues


class TargetsGenerator(object):
    """Compute target values for an objective to minimize.

    Attributes:
        _histories: The reference histories of performance measures.

    """

    def __init__(self):
        self._histories = list()

    def add_history(
            self,
            values_history,  # type: List[float]
            measures_history=None,  # type: Optional[List[float]]
            feasibility_history=None,  # type: Optional[List[bool]]
    ):
        """Add a history of objective values.

        Args:
            values_history: A history of objective values.
                N.B. the value at index i is assumed to have been obtained with i+1
                evaluations.
            measures_history: A history of infeasibility measures.
                If None then measures are set to zero in case of feasibility and set
                to infinity otherwise.
            feasibility_history: A history of feasibilities.
                If None then feasibility is always assumed.

        """
        history = PerformanceHistory(
            values_history, measures_history, feasibility_history
        )
        self._histories.append(history)

    def run(
            self,
            targets_number,  # type: int
            budget_min=1,  # type: Optional[int]
            plot=False  # type: bool
    ):  # type: (...) -> TargetValues
        """Compute target values for a function from histories of its values.

        Args:
            targets_number: The number of targets to compute.
            budget_min: The evaluation budget to be used to define the easiest target.
            plot: Whether to plot the target values.

        Returns:
            The target values of the function.

        """
        # Compute the history of the minimum value
        budget_max = max(len(a_history) for a_history in self._histories)
        minima_histories = [a_hist.cumulated_minimum(fill_up_to=budget_max)
                            for a_hist in self._histories]
        minima_histories = array([a_hist.to_list() for a_hist in minima_histories])
        minimum_history = minima_histories.mean(axis=0)

        # Compute a budget scale
        budget_scale = TargetsGenerator._compute_budget_scale(
            budget_min, budget_max, targets_number
        )

        # Compute the target values
        target_values = minimum_history[budget_scale - 1, :]
        target_values = TargetValues(*zip(*target_values.tolist()))

        # Plot the target values
        if plot:
            objective_values = [inf if a_meas > 0.0 else a_value
                                for a_value, a_meas in target_values]
            self._plot(objective_values)

        return target_values

    @staticmethod
    def _plot(
            objective_target_values  # type: List[float]
    ):
        """Compute and plot the target values.

            Args:
                objective_target_values: The objective target values.

        """
        targets_number = len(objective_target_values)
        figure()
        xlabel("Target index")
        xlim([0, targets_number + 1])
        xticks(linspace(1, targets_number, dtype=int))
        ylabel("Target value")
        semilogy(range(1, targets_number + 1), objective_target_values,
                 marker="o", linestyle="")
        show()

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
