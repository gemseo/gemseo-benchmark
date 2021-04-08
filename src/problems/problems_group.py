from typing import Dict, Iterable, Optional

from data_profiles.data_profile import DataProfile
from problems.bench_problem import BenchProblem


class ProblemsGroup(object):
    """A group of benchmarking problems."""
    # TODO: explain why problems should be grouped.

    def __init__(
            self,
            problems  # type: Iterable[BenchProblem]
    ):
        """
        Args:
            problems: The benchmarking problems of the group.
        """
        self._problems = problems
        # TODO: check that the benchmarking problems have the same number of targets?

    def generate_data_profile(
            self,
            algorithms,  # type: Dict[str, Dict]
            show=True,  # type: bool
            destination_path=None  # type: Optional[str]
    ):  # type: (...) -> None
        """Generate the data profiles of given algorithms relative to the problems.

        Args:
            algorithms: The algorithms and their options.
            show: Whether to show the plot.
            destination_path: The path where to save the plot.
                By default the plot is not saved.
        """
        # Initialize the data profile
        target_values = {
            a_problem.name: a_problem.target_values for a_problem in self._problems
        }
        data_profile = DataProfile({self._name: self._target_values})

