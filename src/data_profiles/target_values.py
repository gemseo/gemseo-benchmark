from numpy import linspace, minimum, ndarray, vstack
from typing import Iterable


class TargetValues(object):
    """Compute target values for an objective to minimize."""

    @staticmethod
    def compute_target_values(values_histories, targets_number, budget_min=1):
        """Compute target values for a function from a history of its values.

        Arguments:
            values_histories (Iterable[List[Number]]): The values history of the
                function.
                The value at index i is assumed to have obtained with i+1 evaluations.
            targets_number (int): The number of targets to compute.
            budget_min (int, optional): The evaluation budget to be used to define the
                easiest target.

        Return:
            (ndarray): Target values for the function.

        """
        # Extend the histories to a common size by repeating their last value
        maximal_size = max(len(a_history) for a_history in values_histories)
        values_histories = [a_hist + [a_hist[-1]] * (maximal_size - len(a_hist))
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
    def _compute_budget_scale(budget_min, budget_max, budgets_number):
        """Compute a scale of evaluation budgets, whose progression relates to
        complexity in terms of evaluation cost.

        N.B. here the evaluation cost is assumed linear with respect to the number of
        evaluations.

        Arguments:
            budget_min (int): the minimum number of evaluations
            budget_max (int): the maximum number of evaluations
            budgets_number (int): the number of budgets

        Returns:
            ndarray: distribution of evaluation budgets

        """
        budget_scale = linspace(budget_min, budget_max, budgets_number, dtype=int)
        return budget_scale
