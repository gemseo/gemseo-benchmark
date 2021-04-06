from typing import Callable, Dict, Iterable, List, Optional, Tuple

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from numpy import ndarray

from data_profiles.target_values import TargetValues
from data_profiles.targets_generator import TargetsGenerator


class BenchProblem(object):
    """A optimization benchmarking problem.

    An optimization benchmarking porblem is characterized by
    - its functions (objective and constraints, including bounds),
    - its starting points,
    - its target values.

    Attributes:
        _creator (Callable): A callable object that returns an instance of the problem.
        _start_points (Iterable[ndarray]): The starting points of the problem.
        _target_values (TargetValues): The target values of the problem.

    """

    def __init__(
            self,
            creator,  # type: Callable[[], OptimizationProblem]
            start_points,  # type: Iterable[ndarray]
            target_values=None,  # type: Optional[TargetValues]
    ):
        self._creator = creator
        self._start_points = start_points
        self._target_values = target_values

    def generate_targets(
            self,
            targets_number,  # type: int
            reference_algorithms,  # type: Dict[str, Dict]
    ):  # type: (...) -> TargetValues
        """Generate targets based on algorithms available in Gemseo.

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
        self._target_values = TargetValues

        return target_values

    @staticmethod
    def _extract_performance(
            problem  # type: OptimizationProblem
    ):  # type: (...) -> Tuple[List[float], List[float], List[bool]]
        """Extract the performance history from a solved Gemseo optimization problem.

        Args:
            problem: The Gemseo optimization problem.

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
