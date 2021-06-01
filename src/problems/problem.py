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

    Attributes:
        name: The name of the benchmarking problem.
        start_points: The starting points of the benchmarking problem.
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
        self.name = name
        self.__creator = creator

        # Set the dimension
        problem = creator()
        if not isinstance(problem, OptimizationProblem):
            raise TypeError("Creator must return an OptimizationProblem")
        self.__dimension = problem.dimension

        # Set the starting points
        if start_points is None:
            if doe_size is None or doe_algo_name is None:
                raise ValueError("The starting points, "
                                 "or their number and the name of the algorithm to "
                                 "generate them, "
                                 "must be passed")
            start_points = self.__generate_start_points(
                doe_algo_name, doe_size, doe_options
            )
        for point in start_points:
            if not isinstance(point, ndarray):
                raise TypeError("Starting points must be of type ndarray")
            if point.shape != (self.__dimension,):
                raise ValueError("Starting points must be 1-dimensional with size {}"
                                 .format(self.__dimension))
        self.start_points = start_points

        self.__target_values = target_values

    def __generate_start_points(
            self,
            doe_algo_name,  # type: str
            doe_size,  # type: int
            doe_options=None,  # type: Optional[Mapping[str, Any]]
    ):  # type: (...) -> Iterable[ndarray]
        """Generate the starting points of the benchmarking problem.

        Args:
            doe_algo_name: The name of the DOE algorithm.
            doe_size: The number of starting points.
            doe_options: The options of the DOE algorithm.

        Returns:
            The starting points of the benchmarking problem.
        """
        design_space = self.__creator().design_space
        doe_library = DOEFactory().create(doe_algo_name)
        if doe_options is None:
            doe_options = dict()
        doe = doe_library(doe_size, design_space.dimension, **doe_options)
        return [design_space.unnormalize_vect(array(row)) for row in doe]

    @property
    def target_values(self):  # type: (...) -> TargetValues
        """The target values of the benchmarking problem."""
        if self.__target_values is None:
            raise ValueError("Benchmarking problem has no target")
        return self.__target_values

    def __iter__(self):  # type: (...) -> OptimizationProblem
        """Iterate on the problem instances with respect to the starting points. """
        for start_point in self.start_points:
            problem = self.__creator()
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
        instance = self.__creator()
        if start_point is not None:
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
        return library.is_algorithm_suited(library.lib_dict[name], self.__creator())

    def generate_targets(
            self,
            targets_number,  # type: int
            ref_algo_specs,  # type: Mapping[str, Mapping[str, Any]]
            feasible=True,  # type: bool
            budget_min=1,  # type: int
            show=False,  # type: bool
            destination_path=None,  # type: Optional[str]
    ):  # type: (...) -> TargetValues
        """Generate targets based on reference algorithms.

        Args:
            targets_number: The number of targets to generate.
            ref_algo_specs: The names and options of the reference algorithms.
            feasible: Whether to generate only feasible targets.
            budget_min: The evaluation budget to be used to define the easiest target.
            show: If True, show the plot.
            destination_path: The path where to save the plot.
                If None, the plot is not saved.

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
        target_values = targets_generator.run(
            targets_number, budget_min, feasible, show, destination_path
        )
        self.__target_values = target_values

        return target_values

    def generate_data_profile(
            self,
            algorithms,  # type: Mapping[str, Mapping[str, Any]]
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
        data_profile = DataProfile({self.name: self.__target_values})

        # Generate the performance histories
        for algo_name, algo_options in algorithms.items():
            for start_point in self.start_points:
                problem = self.get_instance(start_point)
                OptimizersFactory().execute(problem, algo_name, **algo_options)
                obj_values, measures, feas_statuses = self.extract_performance(problem)
                data_profile.add_history(
                    self.name, algo_name, obj_values, measures, feas_statuses
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
            feasibility, measure = problem.get_violation_criteria(key)
            infeas_measures.append(measure)
            feas_statuses.append(feasibility)
        return obj_values, infeas_measures, feas_statuses
