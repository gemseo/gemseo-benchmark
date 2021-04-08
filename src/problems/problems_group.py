from typing import Dict, Iterable, Optional

from gemseo.algos.opt.opt_factory import OptimizersFactory

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
        # TODO: check that every problem has the same number of starting points
        self._problems = problems

    def generate_targets(
            self,
            targets_number,  # type: int
            reference_algorithms,  # type: Dict[str, Dict]
    ):  # type: (...) -> None
        """Generate targets for all the problems based on given reference algorithms.

        Args:
            targets_number: The number of targets to generate.
            reference_algorithms: The names and options of the reference algorithms.
        """
        for a_problem in self._problems:
            a_problem.generate_targets(targets_number, reference_algorithms)

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
        data_profile = DataProfile(target_values)

        # Generate the performance histories
        for an_algo_name, an_algo_options in algorithms.items():
            for a_problem in self._problems:
                for an_instance in a_problem:
                    OptimizersFactory().execute(an_instance, an_algo_name,
                                                **an_algo_options)
                    obj_values, measures, feas = BenchProblem.extract_performance(
                        an_instance
                    )
                    data_profile.add_history(self._name, an_algo_name, obj_values,
                                             measures, feas)
        # TODO: use a "bench"

        # Plot and/or save the data profile
        data_profile.plot(show=show, destination_path=destination_path)
