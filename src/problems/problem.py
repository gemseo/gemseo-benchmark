from typing import Callable, Dict, Iterable, List, Optional, Tuple

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from numpy import ndarray

from data_profiles.data_profile import DataProfile
from data_profiles.target_values import TargetValues
from data_profiles.targets_generator import TargetsGenerator


class Problem(object):
    """An optimization benchmarking problem.

    An optimization benchmarking problem is characterized by
    - its functions (objective and constraints, including bounds),
    - its starting points,
    - its target values.

    """

    def __init__(
            self,
            name,  # type: str
            creator,  # type: Callable[[], OptimizationProblem]
            start_points,  # type: Iterable[ndarray]
            target_values=None,  # type: Optional[TargetValues]
    ):  # type: (...) -> None
        self._name = name
        self._creator = creator
        self._start_points = start_points
        self._target_values = target_values

    def generate_targets(
            self,
            targets_number,  # type: int
            reference_algorithms,  # type: Dict[str, Dict]
    ):  # type: (...) -> TargetValues
        """Generate targets based on reference algorithms.

        Args:
            targets_number: The number of targets to generate.
            reference_algorithms: The names and options of the reference algorithms.

        Returns:
            The generated targets.

        """
        target_generator = TargetsGenerator()

        # Generate reference performance histories
        for an_algo_name, an_algo_options in reference_algorithms.items():
            problem = self._creator()
            OptimizersFactory().execute(problem, an_algo_name, **an_algo_options)
            obj_values, measures, feasibility = self._extract_performance(problem)
            target_generator.add_history(obj_values, measures, feasibility)

        # Compute the target values
        target_values = target_generator.run(targets_number)
        self._target_values = target_values

        return target_values

    def generate_data_profile(
            self,
            algorithms,  # type: Dict[str, Dict]
            show=True,  # type: bool
            destination_path=None  # type: Optional[str]
    ):  # type: (...) -> None
        """Generate a data profile of algorithms available in Gemseo.

        Args:
            algorithms: The algorithms and their options.
            show: Whether to show the plot.
            destination_path: The path where to save the plot.
                (By default the plot is not saved.)

        """
        data_profile = DataProfile({self._name: self._target_values})

        # Generate the performance histories
        for an_algo_name, an_algo_options in algorithms.items():
            problem = self._creator()
            OptimizersFactory().execute(problem, an_algo_name, **an_algo_options)
            obj_values, measures, feasibility = self._extract_performance(problem)
            data_profile.add_history(self._name, an_algo_name, obj_values, measures,
                                     feasibility)

        # Plot and/or save the data profile
        data_profile.plot(show=show, destination_path=destination_path)

    @staticmethod
    def _extract_performance(
            problem  # type: OptimizationProblem
    ):  # type: (...) -> Tuple[List[float], List[float], List[bool]]
        """Extract the performance history from a solved optimization problem.

        Args:
            problem: The optimization problem.

        Returns:
            (
                The history of objective values,
                The history of infeasibility measures,
                The history of feasibility statuses,
            )

        """
        obj_name = problem.objective.name
        history = [(the_values[obj_name],) + problem.get_violation_criteria(an_x)
                   for an_x, the_values in problem.database.items()]
        objective_values, feasibility, infeasibility_measures = zip(*history)
        return objective_values, infeasibility_measures, feasibility
