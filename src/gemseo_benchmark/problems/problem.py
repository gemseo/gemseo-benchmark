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
comparison purposes. A benchmarking problem is characterized by its functions (e.g.
objective and constraints for an optimization problem), its starting points (each
defining an instance of the problem) and its targets (refer to
:mod:`.data_profiles.target_values`).
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Callable
from typing import Union

from gemseo import compute_doe
from gemseo import execute_algo
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.utils.matplotlib_figure import save_show_figure
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import array
from numpy import atleast_2d
from numpy import load
from numpy import ndarray
from numpy import save

from gemseo_benchmark import COLORS_CYCLE
from gemseo_benchmark import MarkeveryType
from gemseo_benchmark import get_markers_cycle
from gemseo_benchmark.data_profiles.data_profile import DataProfile
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.data_profiles.targets_generator import TargetsGenerator
from gemseo_benchmark.results.performance_histories import PerformanceHistories
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.algos.doe.doe_library import DOELibraryOptionType

    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.results.results import Results

InputStartPoints = Union[ndarray, Iterable[ndarray]]


class Problem:
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
        name: str,
        optimization_problem_creator: Callable[[], OptimizationProblem],
        start_points: InputStartPoints | None = None,
        target_values: TargetValues | None = None,
        doe_algo_name: str | None = None,
        doe_size: int | None = None,
        doe_options: Mapping[str, DOELibraryOptionType] | None = None,
        description: str | None = None,
        target_values_algorithms_configurations: AlgorithmsConfigurations | None = None,
        target_values_number: int | None = None,
        optimum: float | None = None,
    ) -> None:
        """
        Args:
            name: The name of the benchmarking problem.
            optimization_problem_creator: A callable object that returns an instance
                of the problem.
            start_points: The starting points of the benchmarking problem.
                If ``None``:
                if ``doe_algo_name``, ``doe_size``, and ``doe_options`` are not ``None``
                then the starting points will be generated as a DOE;
                otherwise the current value of the optimization problem
                will set as the single starting point.
            target_values: The target values of the benchmarking problem.
                If ``None``, the target values will have to be generated later with the
                `generate_targets` method.
            doe_algo_name: The name of the DOE algorithm.
                If ``None``, the current point of the problem design space is set as the
                only starting point.
            doe_size: The number of starting points.
                If ``None``, this number is set as the problem dimension or 10 if
                bigger.
            doe_options: The options of the DOE algorithm.
                If ``None``, no option other than the DOE size is passed to the
                algorithm.
            description: The description of the problem (to appear in a report).
                If ``None``, the problem will not have a description.
            target_values_algorithms_configurations: The configurations of the
                optimization algorithms for the computation of target values.
                If ``None``, the target values will not be computed.
            target_values_number: The number of target values to compute.
                If ``None``, the target values will not be computed.
                N.B. the number of target values shall be the same for all the
                benchmarking problems of a same group.
            optimum: The best feasible objective value of the problem.
                If ``None``, it will not be set.
                If not ``None``, it will be set as the hardest target value.

        Raises:
            TypeError: If the return type of the creator is not
                :class:`gemseo.algos.opt_problem.OptimizationProblem`,
                or if a starting point is not of type ndarray.
            ValueError: If neither starting points nor DOE configurations are passed,
               or if a starting point is of inappropriate shape.
        """  # noqa: D205, D212, D415
        self.name = name
        self.__description = description
        self.creator = optimization_problem_creator
        self.optimum = optimum
        self.__targets_generator = None

        # Set the dimension
        problem = optimization_problem_creator()
        if not isinstance(problem, OptimizationProblem):
            raise TypeError(
                "optimization_problem_creator must return an OptimizationProblem."
            )
        self._problem = problem

        # Set the starting points
        self.__start_points = []
        if start_points is not None:
            self.start_points = start_points
        elif doe_algo_name is not None:
            self.start_points = self.__get_start_points(
                doe_algo_name, doe_size, doe_options
            )
        elif problem.design_space.has_current_value():
            self.start_points = atleast_2d(
                self._problem.design_space.get_current_value()
            )

        # Set the target values:
        self.__target_values = None
        if target_values is not None:
            self.target_values = target_values
        elif (
            target_values_algorithms_configurations is not None
            and target_values_number is not None
        ):
            self.target_values = self.compute_targets(
                target_values_number,
                target_values_algorithms_configurations,
            )

    @property
    def start_points(self) -> list[ndarray]:
        """The starting points of the problem.

        Raises:
            ValueError: If the problem has no starting point,
                or if the starting points are passed as a NumPy array with an invalid
                shape.
        """
        if not self.__start_points:
            raise ValueError("The benchmarking problem has no starting point.")

        return self.__start_points

    @start_points.setter
    def start_points(self, start_points: InputStartPoints) -> None:
        message = (
            "The starting points shall be passed as (lines of) a 2-dimensional "
            "NumPy array, or as an iterable of 1-dimensional NumPy arrays."
        )

        if not isinstance(start_points, ndarray):
            try:
                # try to treat the starting points as an iterable
                iter(start_points)
            except TypeError:
                raise TypeError(
                    f"{message} The following type was passed: {type(start_points)}."
                ) from None

            self.__check_iterable_start_points(start_points)
            start_points_list = list(start_points)

        else:
            # the starting points are passed as a NumPy array
            if start_points.ndim != 2:
                raise ValueError(
                    f"{message} A {start_points.ndim}-dimensional NumPy array "
                    "was passed."
                )

            if start_points.shape[1] != self._problem.dimension:
                raise ValueError(
                    f"{message} The number of columns ({start_points.shape[1]}) "
                    f"is different from the problem dimension "
                    f"({self._problem.dimension})."
                )

            start_points_list = list(start_points)

        # Check that the starting points are within the bounds of the design space
        for point in start_points_list:
            self._problem.design_space.check_membership(point)

        self.__start_points = start_points_list

    def __check_iterable_start_points(self, start_points: Iterable[ndarray]) -> None:
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
            f"{self._problem.dimension}."
        )
        if any(not isinstance(point, ndarray) for point in start_points):
            raise TypeError(error_message)

        if any(
            point.ndim != 1 or point.size != self._problem.dimension
            for point in start_points
        ):
            raise ValueError(error_message)

    def __get_start_points(
        self,
        doe_algo_name: str,
        doe_size: int | None = None,
        doe_options: Mapping[str, DOELibraryOptionType] | None = None,
    ) -> ndarray:
        """Return the starting points of the benchmarking problem.

        Args:
            doe_algo_name: The name of the DOE algorithm.
            doe_size: The number of starting points.
                If ``None``, this number is set as the problem dimension or 10 if
                bigger.
            **doe_options: The options of the DOE algorithm.

        Returns:
            The starting points.
        """
        if doe_size is None:
            doe_size = min([self._problem.dimension, 10])

        if doe_options is None:
            doe_options = {}

        return compute_doe(
            self._problem.design_space, doe_algo_name, doe_size, **doe_options
        )

    @property
    def targets_generator(self) -> TargetsGenerator:
        """The generator for target values."""
        return self.__targets_generator

    @property
    def target_values(self) -> TargetValues:
        """The target values of the benchmarking problem.

        Raises:
            ValueError: If the benchmarking problem has no target value.
        """
        if self.__target_values is None:
            raise ValueError("The benchmarking problem has no target value.")

        return self.__target_values

    @target_values.setter
    def target_values(self, target_values: TargetValues) -> None:
        if not isinstance(target_values, TargetValues):
            raise TypeError(
                f"Target values must be of type TargetValues. "
                f"Type {type(target_values)} was passed."
            )

        self.__target_values = target_values

    def __iter__(self) -> OptimizationProblem:
        """Iterate on the problem instances with respect to the starting points."""
        for start_point in self.start_points:
            problem = self.creator()
            problem.design_space.set_current_value(start_point)
            yield problem

    @property
    def description(self) -> str:
        """The description of the problem."""
        if self.__description is None:
            return "No description available."
        return self.__description

    @property
    def objective_name(self) -> str:
        """The name of the objective function."""
        return self._problem.objective.name

    @property
    def constraints_names(self) -> list[str]:
        """The names of the scalar constraints."""
        return self._problem.get_scalar_constraint_names()

    def is_algorithm_suited(self, name: str) -> bool:
        """Check whether an algorithm is suited to the problem.

        Args:
            name: The name of the algorithm.

        Returns:
            True if the algorithm is suited to the problem, False otherwise.
        """
        library = OptimizersFactory().create(name)
        return library.is_algorithm_suited(library.descriptions[name], self._problem)

    def compute_targets(
        self,
        targets_number: int,
        ref_algo_configurations: AlgorithmsConfigurations,
        only_feasible: bool = True,
        budget_min: int = 1,
        show: bool = False,
        file_path: str | None = None,
        best_target_tolerance: float = 0.0,
        disable_stopping: bool = True,
    ) -> TargetValues:
        """Generate targets based on reference algorithms.

        Args:
            targets_number: The number of targets to generate.
            ref_algo_configurations: The configurations of the reference algorithms.
            only_feasible: Whether to generate only feasible targets.
            budget_min: The evaluation budget to be used to define the easiest target.
            show: If True, show the plot.
            file_path: The path where to save the plot.
                If ``None``, the plot is not saved.
            best_target_tolerance: The relative tolerance for comparisons with the
                best target value.
            disable_stopping: Whether to disable the stopping criteria.

        Returns:
            The generated targets.
        """
        self.__targets_generator = TargetsGenerator()

        # Generate reference performance histories
        for configuration in ref_algo_configurations:
            # Disable the stopping criteria
            options = dict(configuration.algorithm_options)
            if disable_stopping:
                options["xtol_rel"] = 0.0
                options["xtol_abs"] = 0.0
                options["ftol_rel"] = 0.0
                options["ftol_abs"] = 0.0

            for instance in self:
                execute_algo(instance, configuration.algorithm_name, **options)
                history = PerformanceHistory.from_problem(instance)
                self.__targets_generator.add_history(history=history)

        # Compute the target values
        target_values = self.__targets_generator.compute_target_values(
            targets_number,
            budget_min,
            only_feasible,
            show,
            file_path,
            self.optimum,
            best_target_tolerance,
        )
        self.__target_values = target_values

        return target_values

    @staticmethod
    def compute_performance(
        problem: OptimizationProblem,
    ) -> tuple[list[float], list[float], list[bool]]:
        """Extract the performance history from a solved optimization problem.

        Args:
            problem: The optimization problem.

        Returns:
            The history of objective values,
            the history of infeasibility measures,
            the history of feasibility statuses.
        """
        obj_name = problem.objective.name
        obj_values = []
        infeas_measures = []
        feas_statuses = []
        for key, values in problem.database.items():
            obj_values.append(values[obj_name])
            feasibility, measure = problem.get_violation_criteria(key)
            infeas_measures.append(measure)
            feas_statuses.append(feasibility)
        return obj_values, infeas_measures, feas_statuses

    def save_start_points(self, path: Path) -> None:
        """Save the start points as a NumPy binary.

        Args:
            path: The path to the NumPy binary.
        """
        save(path, array(self.start_points))

    def load_start_point(self, path: Path) -> None:
        """Load the start points from a NumPy binary.

        Args:
            path: The path to the NumPy binary.
        """
        self.start_points = load(path)

    @staticmethod
    def _get_description(
        dimension: int,
        nonlinear_objective: bool,
        linear_equality_constraints: int,
        linear_inequality_constraints: int,
        nonlinear_equality_constraints: int,
        nonlinear_inequality_constraints: int,
    ) -> str:
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
            f"A problem depending on {dimension} bounded "
            f"variable{'s' if dimension > 1 else ''}, "
            f"with a {'non' if nonlinear_objective else ''}linear objective"
        )
        if (
            max(
                linear_equality_constraints,
                linear_inequality_constraints,
                nonlinear_equality_constraints,
                nonlinear_inequality_constraints,
            )
            > 0
        ):
            constraints = []
            for number, is_nonlinear, is_inequality in [
                (linear_equality_constraints, False, False),
                (linear_inequality_constraints, False, True),
                (nonlinear_equality_constraints, True, False),
                (nonlinear_inequality_constraints, True, True),
            ]:
                if number > 0:
                    constraints.append(
                        f"{number} {'non' if is_nonlinear else ''}linear "
                        f"{'in' if is_inequality else ''}equality "
                        f"constraint{'s' if number > 1 else ''}"
                    )
            return f"{description}, subject to {', '.join(constraints)}."

        return f"{description}."

    @property
    def dimension(self) -> int:
        """The dimension of the problem."""
        return self._problem.dimension

    def compute_data_profile(
        self,
        algos_configurations: AlgorithmsConfigurations,
        results: Results,
        show: bool = False,
        file_path: str | Path | None = None,
        infeasibility_tolerance: float = 0.0,
        max_eval_number: int | None = None,
    ) -> None:
        """Generate the data profiles of given algorithms.

        Args:
            algos_configurations: The algorithms configurations.
            results: The paths to the reference histories for each algorithm.
            show: Whether to display the plot.
            file_path: The path where to save the plot.
                If ``None``, the plot is not saved.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum evaluations number to be displayed.
                If ``None``, this value is inferred from the longest history.
        """
        # Initialize the data profile
        data_profile = DataProfile({self.name: self.target_values})

        # Generate the performance histories
        for configuration_name in algos_configurations.names:
            for history_path in results.get_paths(configuration_name, self.name):
                history = PerformanceHistory.from_file(history_path)
                if max_eval_number is not None:
                    history = history.shorten(max_eval_number)

                history.apply_infeasibility_tolerance(infeasibility_tolerance)
                data_profile.add_history(
                    self.name,
                    configuration_name,
                    history.objective_values,
                    history.infeasibility_measures,
                )

        # Plot and/or save the data profile
        data_profile.plot(show=show, file_path=file_path)

    def plot_histories(
        self,
        algos_configurations: AlgorithmsConfigurations,
        results: Results,
        show: bool = False,
        file_path: Path | None = None,
        plot_all_histories: bool = False,
        alpha: float = 0.3,
        markevery: MarkeveryType | None = None,
        infeasibility_tolerance: float = 0.0,
        max_eval_number: int | None = None,
        use_log_scale: bool = False,
    ) -> None:
        """Plot the histories of a problem.

        Args:
            algos_configurations: The algorithms configurations.
            results: The paths to the reference histories for each algorithm.
            show: Whether to display the plot.
            file_path: The path where to save the plot.
                If ``None``, the plot is not saved.
            plot_all_histories: Whether to plot all the performance histories.
            alpha: The opacity level for overlapping areas.
                Refer to the Matplotlib documentation.
            markevery: The sampling parameter for the markers of the plot.
                Refer to the Matplotlib documentation.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum evaluations number displayed.
                If ``None``, this value is inferred from the longest history.
            use_log_scale: Whether to use a logarithmic scale on the value axis.
        """
        figure = plt.figure()
        axes = figure.gca()

        # Plot the target values
        objective_targets = [target.objective_value for target in self.target_values]
        for objective_target in objective_targets:
            plt.axhline(objective_target, color="red", linestyle="--")

        # Get the histories of the cumulated minima
        minima, max_feasible_objective = self.__get_cumulated_minimum_histories(
            algos_configurations, results, infeasibility_tolerance, max_eval_number
        )
        if max_eval_number is None:
            max_eval_number = max([
                len(hist) for histories in minima.values() for hist in histories
            ])

        y_relative_margin = 0.03
        max_feasible_objective = self.__get_infeasible_items_objective(
            max_feasible_objective, y_relative_margin
        )

        # Plot the histories
        minimum_values = []
        for configuration_name, color, marker in zip(
            algos_configurations.names, COLORS_CYCLE, get_markers_cycle()
        ):
            minimum_value = minima[configuration_name].plot_algorithm_histories(
                axes,
                configuration_name,
                max_feasible_objective,
                plot_all_histories,
                color=color,
                marker=marker,
                alpha=alpha,
                markevery=markevery,
            )
            if minimum_value is not None:
                minimum_values.append(minimum_value)

        plt.legend()

        # Ensure the x-axis ticks are integers
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        if use_log_scale:
            axes.set_yscale("log")

        plt.margins(x=0.1)
        plt.xlabel("Number of functions evaluations")
        plt.xlim(1, max_eval_number)

        # Set the y-axis margins to zero to get the tight y-limits
        plt.autoscale(enable=True, axis="y", tight=True)
        y_min, y_max = axes.get_ylim()
        # Adjust the y-limits relative to the target values
        if len(objective_targets) > 1:
            y_max = max(*objective_targets, *minimum_values)
            y_min = min(*objective_targets, *minimum_values)
        margin = 0.03 * (y_max - y_min)
        plt.ylim(bottom=y_min - margin, top=y_max + margin)
        plt.ylabel("Objective value")

        # Add ticks for the targets values on a right-side axis
        twin_axes = axes.twinx()
        twin_axes.set_ylim(axes.get_ylim())
        twin_axes.set_yticks(objective_targets)
        twin_axes.set_yticklabels([f"{value:.2g}" for value in objective_targets])
        twin_axes.set_ylabel("Target values", rotation=270)
        if use_log_scale:
            twin_axes.set_yscale("log")

        plt.title("Convergence histories")
        save_show_figure(figure, show, file_path)

    def __get_cumulated_minimum_histories(
        self,
        algos_configurations: AlgorithmsConfigurations,
        results: Results,
        infeasibility_tolerance: float = 0.0,
        max_eval_number: int | None = None,
    ) -> tuple[dict[str, PerformanceHistories], float | None]:
        """Return the histories of the cumulated minimum.

        Args:
            algos_configurations: The algorithms configurations.
            results: The paths to the reference histories for each algorithm.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum evaluations number displayed.
                If ``None``, this value is inferred from the longest history.

        Returns:
            The histories of the cumulated minima and the maximum feasible value.
        """
        minima = {}
        max_feasible_objective = None
        for configuration_name in algos_configurations.names:
            minima[configuration_name] = PerformanceHistories()
            for path in results.get_paths(configuration_name, self.name):
                # Get the history of the cumulated minimum
                history = PerformanceHistory.from_file(path)
                if max_eval_number is not None:
                    history = history.shorten(max_eval_number)

                history.apply_infeasibility_tolerance(infeasibility_tolerance)
                history = history.compute_cumulated_minimum()
                minima[configuration_name].append(history)

                # Update the maximum feasible objective value
                feasible_objectives = [
                    item.objective_value for item in history if item.is_feasible
                ]
                if max_feasible_objective is None:
                    max_feasible_objective = max(feasible_objectives, default=None)
                else:
                    max_feasible_objective = max([
                        *feasible_objectives,
                        max_feasible_objective,
                    ])

        return minima, max_feasible_objective

    def __get_infeasible_items_objective(
        self, max_feasible_objective: float | None, y_relative_margin: float
    ) -> float:
        """Return the objective value to set for the infeasible history items.

        This finite value will serve for the graph of the maximum history.

        Args:
            max_feasible_objective: The maximum feasible objective value.
                None means there is no feasible objective value.
            y_relative_margin: The vertical relative margin for the histories plot.

        Returns:
            The objective value to set for the infeasible history items.
        """
        objective_targets = [target.objective_value for target in self.target_values]
        if max_feasible_objective is None:
            max_feasible_objective = max(objective_targets)
        else:
            max_feasible_objective = max(max_feasible_objective, *objective_targets)

        if self.optimum is None:
            return max_feasible_objective

        return max_feasible_objective + y_relative_margin * (
            max_feasible_objective - self.optimum
        )
