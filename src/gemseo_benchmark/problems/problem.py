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
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Union

from numpy import array, atleast_2d, load, ndarray, save

from gemseo import api
from gemseo.algos.doe.doe_lib import DOELibraryOptionType
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import compute_doe
from gemseo.utils.py23_compat import Path
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.data_profiles.targets_generator import TargetsGenerator
from gemseo_benchmark.results.performance_history import PerformanceHistory
from gemseo_benchmark.utils import get_scalar_constraints_names

InputStartPoints = Union[ndarray, Iterable[ndarray]]
AlgorithmsSpecifications = Mapping[str, Mapping[str, Any]]


class Problem(object):
    """An optimization benchmarking problem.

    An optimization benchmarking problem is characterized by
    - its functions (objective and constraints, including bounds),
    - its starting points,
    - its target values.

    Attributes:
        name (str): The name of the benchmarking problem.
        optimization_problem_creator (Callable[[], OptimizationProblem]): A callable
        that returns an instance of the optimization problem.
        start_points (Iterable[ndarray]): The starting points of the benchmarking
            problem.
        optimum (float): The best feasible objective value of the problem.
            Set to None if unknown.
    """

    def __init__(
            self,
            name,  # type: str
            optimization_problem_creator,  # type: Callable[[], OptimizationProblem]
            start_points=None,  # type: Optional[InputStartPoints]
            target_values=None,  # type: Optional[TargetValues]
            doe_algo_name=None,  # type: Optional[str]
            doe_size=None,  # type: Optional[int]
            doe_options=None,  # type: Optional[Mapping[str, DOELibraryOptionType]]
            description=None,  # type: Optional[str]
            target_values_algorithms=None,  # type: Optional[AlgorithmsSpecifications]
            target_values_number=None,  # type: Optional[int]
            optimum=None,  # type: Optional[float]
    ):  # type: (...) -> None
        """
        Args:
            name: The name of the benchmarking problem.
            optimization_problem_creator: A callable object that returns an instance
                of the problem.
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
            description: The description of the problem (to appear in a report).
                If None, the problem will not have a description.
            target_values_algorithms: The specifications of the optimization algorithms
                for the computation of target values.
                If None, the target values will not be computed.
            target_values_number: The number of target values to compute.
                If None, the target values will not be computed.
                N.B. the number of target values shall be the same for all the
                benchmarking problems of a same group.
            optimum: The best feasible objective value of the problem.
                If None, it will not be set.
                If not None, it will be set as the hardest target value.

        Raises:
            TypeError: If the return type of the creator is not
                :class:`.OptimizationProblem`,
                or if a starting point is not of type ndarray.
            ValueError: If neither starting points nor DOE specifications are passed,
               or if a starting point is of inappropriate shape.
        """
        self.name = name
        self.__description = description
        self.creator = optimization_problem_creator
        self.optimum = optimum
        self.__targets_generator = None

        # Set the dimension
        problem = optimization_problem_creator()
        if not isinstance(problem, OptimizationProblem):
            raise TypeError("optimization_problem_creator must return an OptimizationProblem.")
        self.__problem = problem

        # Set the starting points
        self.__start_points = list()
        if start_points is not None:
            self.start_points = start_points
        elif doe_algo_name is not None:
            self.start_points = self.__get_start_points(
                doe_algo_name, doe_size, doe_options
            )
        elif problem.design_space.has_current_x():
            self.start_points = atleast_2d(self.__problem.design_space.get_current_x())

        # Set the target values:
        self.__target_values = None
        if target_values is not None:
            self.target_values = target_values
        elif target_values_algorithms is not None and target_values_number is not None:
            self.target_values = self.compute_targets(
                target_values_number, target_values_algorithms,
            )

    @property
    def start_points(self):  # type: (...) -> List[ndarray]
        """The starting points of the problem.

        Raises:
            ValueError: If the problem has no starting point,
                or if the starting points are passed as a NumPy array with an invalid
                shape.
            TypeError: If the starting points are passed neither as a NumPy array
                nor as an iterable.
        """
        if not self.__start_points:
            raise ValueError("The benchmarking problem has no starting point.")

        return self.__start_points

    @start_points.setter
    def start_points(
            self,
            start_points,  # type: InputStartPoints
    ):  # type: (...) -> None
        message = "The starting points shall be passed as (lines of) a 2-dimensional " \
                  "NumPy array, or as an iterable of 1-dimensional NumPy arrays."

        if not isinstance(start_points, ndarray):
            try:
                # try to treat the starting points as an iterable
                iter(start_points)
            except TypeError:
                raise TypeError(
                    "{} The following type was passed: {}.".format(
                        message, type(start_points)
                    )
                )

            self.__check_iterable_start_points(start_points)
            start_points_list = list(start_points)

        else:
            # the starting points are passed as a NumPy array
            if start_points.ndim != 2:
                raise ValueError(
                    "{} A {}-dimensional NumPy array was passed.".format(
                        message, start_points.ndim
                    )
                )

            if start_points.shape[1] != self.__problem.dimension:
                raise ValueError(
                    "{} The number of columns ({}) is different from the problem "
                    "dimension ({}).".format(
                        message, start_points.shape[1], self.__problem.dimension
                    )
                )

            start_points_list = [point for point in start_points]

        # Check that the starting points are within the bounds of the design space
        for point in start_points_list:
            self.__problem.design_space.check_membership(point)

        self.__start_points = start_points_list

    def __check_iterable_start_points(
            self,
            start_points,  # type: Iterable[ndarray]
    ):  # type: (...) -> None
        """Check starting points passed as an iterable.

        Args:
            start_points: The starting points.

        Raises:
            TypeError: If the iterable contains at least one item that is not a NumPy
                array.
            ValueError: If the iterable contains NumPy arrays of the wrong shape.
        """
        error_message = (
            "A starting point must be a 1-dimensional NumPy array of size "
            "{}.".format(self.__problem.dimension)
        )
        if any(not isinstance(point, ndarray) for point in start_points):
            raise TypeError(error_message)

        if any(point.ndim != 1 or point.size != self.__problem.dimension for point in start_points):
            raise ValueError(error_message)

    def __get_start_points(
            self,
            doe_algo_name,  # type: str
            doe_size=None,  # type: Optional[int]
            doe_options=None  # type: Optional[Mapping[str, DOELibraryOptionType]]
    ):  # type: (...) -> ndarray
        """Return the starting points of the benchmarking problem.

        Args:
            doe_algo_name: The name of the DOE algorithm.
            doe_size: The number of starting points.
                If None, this number is set as the problem dimension or 10 if bigger.
            **doe_options: The options of the DOE algorithm.

        Returns:
            The starting points.
        """
        if doe_size is None:
            doe_size = min([self.__problem.dimension, 10])

        if doe_options is None:
            doe_options = dict()

        return compute_doe(
            self.__problem.design_space, doe_algo_name, doe_size, **doe_options
        )

    @property
    def targets_generator(self):  # type: (...) -> TargetsGenerator
        """The generator for target values."""
        return self.__targets_generator

    @property
    def target_values(self):  # type: (...) -> TargetValues
        """The target values of the benchmarking problem.

        Raises:
            ValueError: If the benchmarking problem has no target value.
        """
        if self.__target_values is None:
            raise ValueError("The benchmarking problem has no target value.")

        return self.__target_values

    @target_values.setter
    def target_values(
            self,
            target_values  # type: TargetValues
    ):
        if not isinstance(target_values, TargetValues):
            raise TypeError
        self.__target_values = target_values

    def __iter__(self):  # type: (...) -> OptimizationProblem
        """Iterate on the problem instances with respect to the starting points. """
        for start_point in self.start_points:
            problem = self.creator()
            problem.design_space.set_current_x(start_point)
            yield problem

    @property
    def description(self):  # type: (...) -> str
        """The description of the problem."""
        if self.__description is None:
            return "No description available."
        return self.__description

    @property
    def objective_name(self):  # type: (...) -> str
        """The name of the objective function."""
        return self.__problem.objective.name

    @property
    def constraints_names(self):  # type: (...) -> List[str]
        """The names of the scalar constraints."""
        return get_scalar_constraints_names(self.__problem)

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
            ref_algo_specifications,  # type: AlgorithmsSpecifications
            only_feasible=True,  # type: bool
            budget_min=1,  # type: int
            show=False,  # type: bool
            path=None,  # type: Optional[str]
            best_target_tolerance=0.0,  # type: float
            disable_stopping=True  # type: bool
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
            best_target_tolerance: The relative tolerance for comparisons with the
                best target value.
            disable_stopping: Whether to disable the stopping criteria.

        Returns:
            The generated targets.
        """
        self.__targets_generator = TargetsGenerator()

        # Generate reference performance histories
        for algo_name, algo_options in ref_algo_specifications.items():
            # Disable the stopping criteria
            options = dict(algo_options)
            if disable_stopping:
                options["xtol_rel"] = 0.0
                options["xtol_abs"] = 0.0
                options["ftol_rel"] = 0.0
                options["ftol_abs"] = 0.0

            for instance in self:
                api.execute_algo(instance, algo_name, **options)
                history = PerformanceHistory.from_problem(instance)
                self.__targets_generator.add_history(history=history)

        # Compute the target values
        target_values = self.__targets_generator.compute_target_values(
            targets_number, budget_min, only_feasible, show, path, self.optimum,
            best_target_tolerance
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

    def save_start_points(
            self,
            path,  # type: Path
    ):  # type: (...) -> None
        """Save the start points as a NumPy binary.

        Args:
            path: The path to the NumPy binary.
        """
        save(path, array(self.start_points))

    def load_start_point(
            self,
            path,  # type: Path
    ):  # type: (...) -> None
        """Load the start points from a NumPy binary.

        Args:
            path: The path to the NumPy binary.
        """
        self.start_points = load(path)

    @staticmethod
    def _get_description(
            dimension,  # type: int
            nonlinear_objective,  # type: bool
            linear_equality_constraints,  # type: int
            linear_inequality_constraints,  # type: int
            nonlinear_equality_constraints,  # type: int
            nonlinear_inequality_constraints,  # type: int
    ):  # type: (...) -> str
        """Return a formal description of the problem.

        Args:
            dimension: The number of optimization variables.
            nonlinear_objective: Whether the objective is nonlinear.
            linear_equality_constraints: The number of linear equality constraints.
            linear_inequality_constraints: The number of linear inequality constraints.
            nonlinear_equality_constraints: The number of nonlinear equality
                constraints.
            nonlinear_inequality_constraints: The number of nonlinear inequality
                constraints.

        Returns:
            The description of the problem.
        """
        description = (
            "A problem depending on {} bounded variable{}, "
            "with a {}linear objective".format(
                dimension, "s" if dimension > 1 else "",
                "non" if nonlinear_objective else ""
            )
        )
        if max(
                linear_equality_constraints,
                linear_inequality_constraints,
                nonlinear_equality_constraints,
                nonlinear_inequality_constraints
        ) > 0:
            constraints = []
            for number, is_nonlinear, is_inequality in [
                (linear_equality_constraints, False, False),
                (linear_inequality_constraints, False, True),
                (nonlinear_equality_constraints, True, False),
                (nonlinear_inequality_constraints, True, True)
            ]:
                if number > 0:
                    constraints.append(
                        "{} {}linear {}equality constraint{}".format(
                            number,
                            "non" if is_nonlinear else "",
                            "in" if is_inequality else "",
                            "s" if number > 1 else ""
                        )
                    )
            return "{}, subject to {}.".format(description, ", ".join(constraints))

        return "{}.".format(description)

    @property
    def dimension(self):  # type: (...) -> int
        """The dimension of the problem."""
        return self.__problem.dimension
