# -*- coding: utf-8 -*-
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
"""Reference problem for benchmarking.

A benchmarking problem is a problem class to be solved by iterative algorithms for
comparison purposes.
A benchmarking problem is characterized by its functions
(e.g. objective and constraints for an optimization problem),
its starting points (each defining an instance of the problem)
and its targets (refer to :mod:`target_values`).
"""
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import compute_doe
from numpy import ndarray

from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.data_profiles.targets_generator import TargetsGenerator
from gemseo_benchmark.utils import get_scalar_constraints_names


class Problem(object):
    """An optimization benchmarking problem.

    An optimization benchmarking problem is characterized by
    - its functions (objective and constraints, including bounds),
    - its starting points,
    - its target values.

    Attributes:
        name (str): The name of the benchmarking problem.
        creator (Callable[[], OptimizationProblem]): A callable that returns an instance
            of the optimization problem.
        start_points (Iterable[ndarray]): The starting points of the benchmarking
            problem.
        objective_name (str): The name of the objective function.
        constraints_names (List[str]): The names of the scalar constraints.
    """

    def __init__(
            self,
            name,  # type: str
            creator,  # type: Callable[[], OptimizationProblem]
            start_points=None,  # type: Optional[Iterable[ndarray]]
            target_values=None,  # type: Optional[TargetValues]
            doe_algo_name=None,  # type: Optional[str]
            doe_size=None,  # type: Optional[int]
            doe_options=None,  # type: Optional[Dict[str, Any]]
    ):  # type: (...) -> None
        """
        Args:
            name: The name of the benchmarking problem.
            creator: A callable object that returns an instance of the problem.
            start_points: The starting points of the benchmarking problem.
                If None, the start points will generated as a DOE.
            target_values: The target values of the benchmarking problem.
                If None, the target values will have to be generated later with the
                `generate_targets` method.
            doe_algo_name: The name of the DOE algorithm.
                If None, the current point of the problem design space is set as the
                only starting point.
            doe_size: The number of starting points.
                If None, this number is set as the problem dimension or 10 if bigger.
            doe_options: The options of the DOE algorithm.
                If None, no option other than the DOE size is passed to the algorithm.

        Raises:
            TypeError: If the return type of the creator is not
                :class:`.OptimizationProblem`,
                or if a starting point is not of type ndarray.
            ValueError: If neither starting points nor DOE specifications are passed,
               or if a starting point is of inappropriate shape.
        """
        self.name = name
        self.creator = creator

        # Set the dimension
        problem = creator()
        if not isinstance(problem, OptimizationProblem):
            raise TypeError("Creator must return an OptimizationProblem.")
        self.__problem = problem

        self.__dimension = problem.dimension

        # Set the functions names
        self.objective_name = problem.objective.name
        self.constraints_names = get_scalar_constraints_names(problem)

        # Set the starting points
        if start_points is None:
            if doe_options is None:
                doe_options = dict()
            self.start_points = self.__get_start_points(
                doe_algo_name, doe_size, **doe_options
            )
        else:
            self.start_points = start_points

        self.__check_start_points()
        self.__target_values = target_values

    def __get_start_points(
            self,
            doe_algo_name=None,  # type: Optional[str]
            doe_size=None,  # type: Optional[int]
            **doe_options,  # type: Any
    ):  # type: (...) -> Iterable[ndarray]
        """Return the starting points of the benchmarking problem.

        Args:
            doe_algo_name: The name of the DOE algorithm.
                If None, the current point of the problem design space is set as the
                only starting point.
            doe_size: The number of starting points.
                If None, this number is set as the problem dimension or 10 if bigger.
            **doe_options: The options of the DOE algorithm.

        Returns:
            The starting points.

        Raises:
            ValueError: If no DOE algorithm name is specified
                and the problem has no current point.
        """
        if doe_algo_name is not None:
            if doe_size is None:
                doe_size = min([self.__dimension, 10])

            return compute_doe(
                self.__problem.design_space, doe_size, doe_algo_name, **doe_options
            )

        # Set the current point of the design space as single starting point.
        if not self.__problem.design_space.has_current_x():
            raise ValueError(
                "The problem has neither DOE algorithm name"
                "nor current point"
                "to set the starting points."
            )

        return [self.__problem.design_space.get_current_x()]

    def __check_start_points(self):  # type: (...) -> None
        """Check the starting points of the benchmarking problem.

        Raises:
            TypeError: If a starting point is not of type ndarray.
            ValueError: If a starting point is of inappropriate shape.
        """
        for point in self.start_points:
            if not isinstance(point, ndarray):
                raise TypeError(
                    "The starting points must be of type ndarray."
                    " The following type was passed: {}.".format(type(point))
                )

            if point.shape != (self.__dimension,):
                raise ValueError(
                    "Starting points must be 1-dimensional with size {}."
                    " The following shape was passed: {}.".format(
                        self.__dimension, point.shape
                    )
                )

    @property
    def target_values(self):  # type: (...) -> TargetValues
        """The target values of the benchmarking problem."""
        if self.__target_values is None:
            raise ValueError("Benchmarking problem has no target")
        return self.__target_values

    def __iter__(self):  # type: (...) -> OptimizationProblem
        """Iterate on the problem instances with respect to the starting points. """
        for start_point in self.start_points:
            problem = self.creator()
            problem.design_space.set_current_x(start_point)
            yield problem

    def is_algorithm_suited(
            self,
            name,  # type: str
    ):  # type: (...) -> bool
        """Check whether an algorithm is suited to the problem.

        Args:
            name: The name of the algorithm.

        Returns:
            True if the algorithm is suited to the problem, False otherwise.
        """
        library = OptimizersFactory().create(name)
        return library.is_algorithm_suited(library.lib_dict[name], self.__problem)

    def compute_targets(
            self,
            targets_number,  # type: int
            ref_algo_specifications,  # type: Mapping[str, Mapping[str, Any]]
            only_feasible=True,  # type: bool
            budget_min=1,  # type: int
            show=False,  # type: bool
            path=None,  # type: Optional[str]
    ):  # type: (...) -> TargetValues
        """Generate targets based on reference algorithms.

        Args:
            targets_number: The number of targets to generate.
            ref_algo_specifications: The names and options of the reference algorithms.
            only_feasible: Whether to generate only feasible targets.
            budget_min: The evaluation budget to be used to define the easiest target.
            show: If True, show the plot.
            path: The path where to save the plot.
                If None, the plot is not saved.

        Returns:
            The generated targets.
        """
        targets_generator = TargetsGenerator()

        # Generate reference performance histories
        for algo_name, algo_options in ref_algo_specifications.items():
            for instance in self:
                OptimizersFactory().execute(instance, algo_name, **algo_options)
                obj_values, measures, feas_statuses = self.compute_performance(instance)
                targets_generator.add_history(obj_values, measures, feas_statuses)

        # Compute the target values
        target_values = targets_generator.run(
            targets_number, budget_min, only_feasible, show, path
        )
        self.__target_values = target_values

        return target_values

    @staticmethod
    def compute_performance(
            problem  # type: OptimizationProblem
    ):  # type: (...) -> Tuple[List[float], List[float], List[bool]]
        """Extract the performance history from a solved optimization problem.

        Args:
            problem: The optimization problem.

        Returns:
            The history of objective values,
            the history of infeasibility measures,
            the history of feasibility statuses.
        """
        obj_name = problem.objective.name
        obj_values = list()
        infeas_measures = list()
        feas_statuses = list()
        for key, values in problem.database.items():
            obj_values.append(values[obj_name])
            feasibility, measure = problem.get_violation_criteria(key)
            infeas_measures.append(measure)
            feas_statuses.append(feasibility)
        return obj_values, infeas_measures, feas_statuses
