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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from numpy import array, ndarray

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
            target_values: The target values of the benchmarking problem.
                If None, the target values will have to be generated later with the
                `generate_targets` method.
            doe_algo_name: The name of the DOE algorithm.
            doe_size: The number of starting points.
            doe_options: The options of the DOE algorithm.

        Raises:
            TypeError: If the return type of the creator is not OptimizationProblem,
                or if a starting point is not of type ndarray.
            ValueError: If neither starting points nor DOE specifications are passed,
               or if a starting point is of inappropriate shape.
        """
        self._name = name
        self._creator = creator

        # Set the dimension
        problem = creator()
        if not isinstance(problem, OptimizationProblem):
            raise TypeError("Creator must return an OptimizationProblem")
        self._dimension = problem.dimension

        # Set the starting points
        if start_points is None and (doe_size is None or doe_algo_name is None):
            raise ValueError("The starting points, "
                             "or their number and the name of the algorithm to "
                             "generate them, "
                             "must be passed")
        elif start_points is None:
            start_points = self._generate_start_points(
                doe_size, doe_algo_name, doe_options
            )
        for point in start_points:
            if not isinstance(point, ndarray):
                raise TypeError("Starting points must be of type ndarray")
            elif point.shape != (self._dimension,):
                raise ValueError("Starting points must be 1-dimensional with size {}"
                                 .format(self._dimension))
        self._start_points = start_points

        self._target_values = target_values

    def _generate_start_points(
            self,
            doe_algo_name,  # type: str
            doe_size,  # type: int
            doe_options=None,  # type: Optional[Dict[str, Any]]
    ):  # type: (...) -> Iterable[ndarray]
        """Generate the starting points of the benchmarking problem.

        Args:
            doe_algo_name: The name of the DOE algorithm.
            doe_size: The number of starting points.
            doe_options: The options of the DOE algorithm.

        Returns:
            The starting points of the benchmarking problem.
        """
        design_space = self._creator().design_space
        doe_library = DOEFactory().create(doe_algo_name)
        doe = doe_library(doe_size, design_space.dimension, **doe_options)
        return [design_space.unnormalize_vect(array(row)) for row in doe]

    @property
    def name(self):  # type: (...) -> str
        """The name of the benchmarking problem."""
        return self._name

    @property
    def start_points(self):  # type: (...) -> Iterable[ndarray]
        """The starting points of the benchmarking problem."""
        return self._start_points

    @property
    def target_values(self):  # type: (...) -> TargetValues
        """The target values of the benchmarking problem."""
        if self._target_values is None:
            raise ValueError("Benchmarking problem has no target")
        return self._target_values

    def __iter__(self):  # type: (...) -> OptimizationProblem
        """Iterate on the problem instances with respect to the starting points. """
        for start_point in self._start_points:
            problem = self._creator()
            problem.design_space.set_current_x(start_point)
            yield problem

    def get_instance(
            self,
            start_point=None  # type: Optional[ndarray]
    ):  # type: (...) -> OptimizationProblem
        """Return an instance of the benchmarking problem.

        Args:
            start_point: The starting point of the instance.
                If None, it is the current design of the benchmarking problem.

        Returns:
            The instance of the benchmarking problem.
        """
        # TODO: remove this method
        instance = self._creator()
        if start_point is not None and start_point in self._start_points:
            instance.design_space.set_current_x(start_point)
        return instance

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
        return library.is_algorithm_suited(library.lib_dict[name], self._creator())

    def generate_targets(
            self,
            targets_number,  # type: int
            ref_algo_specs,  # type: Dict[str, Dict[str, Any]]
            feasible=True,  # type: bool
    ):  # type: (...) -> TargetValues
        """Generate targets based on reference algorithms.

        Args:
            targets_number: The number of targets to generate.
            ref_algo_specs: The names and options of the reference algorithms.
            feasible: Whether to generate only feasible targets.

        Returns:
            The generated targets.
        """
        targets_generator = TargetsGenerator()

        # Generate reference performance histories
        for algo_name, algo_options in ref_algo_specs.items():
            for instance in self:
                OptimizersFactory().execute(instance, algo_name, **algo_options)
                obj_values, measures, feas_statuses = self.extract_performance(instance)
                targets_generator.add_history(obj_values, measures, feas_statuses)

        # Compute the target values
        target_values = targets_generator.run(targets_number, feasible=feasible)
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
                If None, the plot is not saved.
        """
        data_profile = DataProfile({self._name: self._target_values})

        # Generate the performance histories
        for algo_name, algo_options in algorithms.items():
            for start_point in self._start_points:
                problem = self.get_instance(start_point)
                OptimizersFactory().execute(problem, algo_name, **algo_options)
                obj_values, measures, feas_statuses = self.extract_performance(problem)
                data_profile.add_history(
                    self._name, algo_name, obj_values, measures, feas_statuses
                )

        # Plot and/or save the data profile
        data_profile.plot(show=show, destination_path=destination_path)

    # TODO: remove this method (use ProblemsGroup)

    @staticmethod
    def extract_performance(
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
            measure, feasibility = problem.get_violation_criteria(key)
            infeas_measures.append(measure)
            feas_statuses.append(feasibility)
        return obj_values, infeas_measures, feas_statuses
