from itertools import chain, repeat

from numbers import Number
from numpy import inf, linspace, minimum, ndarray, vstack
from typing import Iterable, List, Optional


class TargetValues(object):
    """Compute target values for an objective to minimize."""

    @staticmethod
    def compute_target_values(
            values_histories,  # type: Iterable[List[Number]]
            targets_number,  # type: int
            budget_min=1,  # type: Optional[int]
            feasibility_histories=None,  # type: Optional[Iterable[List[bool]]]
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

        Returns:
            Target values for the function.

        """
        # Check the histories lengths
        if feasibility_histories is None:
            feasibility_histories = [[True] * len(a_history) for a_history
                                     in values_histories]
        for a_val_hist, a_feas_hist in zip(values_histories, feasibility_histories):
            if len(a_val_hist) != len(a_feas_hist):
                raise ValueError("Histories of values and feasibility must have same "
                                 "size")

        # Take feasibility into account
        values_histories = [a_value if a_feasibility else inf
                            for a_value, a_feasibility in zip(values_histories,
                                                              feasibility_histories)]
        # TODO: use a feasibility measure / penalty

        # Extend the histories to a common size by repeating their last value
        maximal_size = max(len(a_history) for a_history in values_histories)
        values_histories = [chain(a_hist,
                                  repeat(a_hist[-1], (maximal_size - len(a_hist))))
                            for a_hist in values_histories]

        # Compute the history of the minimum value
        minimum_history = vstack([minimum.accumulate(a_hist) for a_hist in
                                  values_histories])
        minimum_history = minimum_history.mean(axis=0)

        # Compute a budget scale
        budget_max = len(minimum_history)
        budget_scale = TargetValues._compute_budget_scale(budget_min, budget_max,
                                                          targets_number)

        # Compute the target values
        target_values = minimum_history[budget_scale - 1]

        return target_values

    @staticmethod
    def _compute_budget_scale(budget_min,  # type: int
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
