from numbers import Number
from typing import Dict, Iterable, List, Optional

from matplotlib import rcParams
from matplotlib.pyplot import (axhline, figure, legend, plot, show, title, xlabel,
                               xlim, ylabel, ylim, yticks)
from numpy import append, array, linspace, zeros

from data_profiles.target_values import TargetValues
from data_profiles.performance_history import PerformanceHistory


class DataProfile(object):
    """Compute the data profiles of optimizers on optimization problems.

    Attributes:
        _target_values: The target values of each of the reference problems.
        _targets_number: The number of target values.
        _values_histories: The histories of values of the compared algorithm for each of
            the reference problems.

    """

    def __init__(
            self,
            target_values  # type: Dict[str, TargetValues]
    ):
        self._target_values = None
        self._targets_number = 0
        self.target_values = target_values
        self._values_histories = dict()

    @property
    def target_values(self):
        """The target values of each reference problem."""
        return self._target_values

    @target_values.setter
    def target_values(
            self,
            target_values  # type: Dict[str, TargetValues]
    ):
        if not isinstance(target_values, dict):
            raise TypeError("The target values be must passed as a dictionary")
        targets_numbers = set(len(pb_targets) for pb_targets in target_values.values())
        if len(targets_numbers) != 1:
            raise ValueError("The reference problems must have the same number of "
                             "target values")
        self._target_values = target_values
        self._targets_number = targets_numbers.pop()

    def add_history(
            self,
            problem_name,  # type: str
            algo_name,  # type: str
            values_history,  # type: List[float]
            measures_history=None,  # type: Optional[List[float]]
            feasibility_history=None,  # type: Optional[List[bool]]
    ):
        """Add a history of performance values.

        Args:
            problem_name: The name of the problem.
            algo_name: The name of the algorithm.
            values_history: A history of objective values.
                N.B. the value at index i is assumed to have been obtained with i+1
                evaluations.
            measures_history: A history of infeasibility measures.
                If None then measures are set to zero in case of feasibility and set
                to infinity otherwise.
            feasibility_history: A history of feasibilities.
                If None then feasibility is always assumed.

        """
        if problem_name not in self._target_values:
            raise ValueError(
                "{} is not the name of a reference problem".format(problem_name)
            )
        if algo_name not in self._values_histories:
            self._values_histories[algo_name] = {
                a_pb_name: list() for a_pb_name in self._target_values.keys()
            }
        history = PerformanceHistory(
            values_history, measures_history, feasibility_history
        )
        self._values_histories[algo_name][problem_name].append(history)

    def plot(
            self,
            algo_names=None  # type: Optional[Iterable[str]]
    ):
        """Plot the data profiles of the required algorithms.

        Args:
            algo_names: The names of the algorithms.
                If None then all the algorithms are considered.
        """
        data_profiles = self.compute_data_profiles(algo_names)
        DataProfile.plot_data_profile(data_profiles)

    def compute_data_profiles(
            self,
            algo_names=None  # type: Optional[Iterable[str]]
    ):  # type: (...) -> Dict[str, List[Number]]
        """Compute the data profiles for the required algorithms relative to the
        reference problems.

        Args:
            algo_names: The names of the algorithms.
                If None then all the algorithms are considered.

        Returns:
            The data profiles.

        """
        algo_names = self._values_histories.keys() if algo_names is None else algo_names
        data_profiles = {a_name: self.compute_a_data_profile(a_name)
                         for a_name in algo_names}
        return data_profiles

    def compute_a_data_profile(
            self,
            algo_name  # type: str
    ):  # type: (...) -> List[Number]
        """Compute the history of the target hits ratios associated with an optimizer
        (with respect to the target values of optimization problems).

        Args:
            algo_name: The name of the algorithm.

        Returns:
            The history of the success ratios.

        """
        repeat_number = self._check_repeat_number(algo_name)
        algo_histories = self._values_histories[algo_name]

        # Compute the history of total target hits
        max_history_size = max([max([len(a_pb_hist) for a_pb_hist in a_pb_histories])
                                for a_pb_histories in algo_histories.values()])
        total_hits_history = zeros(max_history_size)
        for a_pb_name, targets in self._target_values.items():
            for a_pb_history in algo_histories[a_pb_name]:
                hits_history = targets.count_targets_hits(a_pb_history)
                # If the history is shorter than the longest one, repeat its last value
                if len(hits_history) < max_history_size:
                    tail = [hits_history[-1]] * (max_history_size - hits_history.size)
                    hits_history = hits_history.extend(tail)
                total_hits_history += array(hits_history)

        # Return the history of target hits ratios
        problems_number = len(self._target_values)
        targets_total = self._targets_number * problems_number * repeat_number
        ratios = total_hits_history / targets_total
        return ratios.tolist()

    def _check_repeat_number(
            self,
            algo_name  # type: str
    ):  # type: (...) -> int
        """Check that an algorithm has the same number of values histories for each
        of the reference problems (so that the reference problems are equally
        represented with respect to the algorithm performance).

        Args:
            algo_name: The name of the algorithm.

        Returns:
            The common number of values histories per problem.

        """
        histories_numbers = set(len(histories) for histories
                                in self._values_histories[algo_name].values())
        if len(histories_numbers) != 1:
            raise ValueError("Reference problems unequally represented for algorithm {}"
                             .format(algo_name))
        return histories_numbers.pop()

    @staticmethod
    def plot_data_profile(data_profiles  # type: Dict[str, List[Number]]
                          ):
        """Plot data profiles.

        Args:
            data_profiles: The data profiles.

        """
        figure()

        # Set the title and axes
        title("Data profile{}".format("s" if len(data_profiles) > 1 else ""))
        max_profile_size = max([len(a_profile) for a_profile in data_profiles.values()])
        xlabel("Number of functions evaluations")
        xlim([1, max_profile_size])
        y_ticks = linspace(0.0, 1.0, 11)
        yticks(y_ticks, ("{:02.0f}%".format(ratio * 100.0) for ratio in y_ticks))
        ylabel("Ratios of targets reached")
        ylim([0.0, 1.05])

        # Plot the 100% line
        axhline(1.0, linestyle=":", color="black")

        # Plot the data profiles
        color_cycle = rcParams["axes.prop_cycle"].by_key()["color"]
        for a_color, (a_name, a_profile) in zip(color_cycle, data_profiles.items()):
            last_abscissa = len(a_profile)
            last_value = a_profile[-1]
            # Extend the profile if necessary
            if last_abscissa < max_profile_size:
                tail = [last_value] * (max_profile_size - last_abscissa)
                a_profile = append(a_profile, tail)
            plot(range(1, max_profile_size + 1), a_profile, color=a_color, label=a_name)
            plot(last_abscissa + 1, last_value, marker="*")

        legend()
        show()
