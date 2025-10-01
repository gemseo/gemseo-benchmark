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
"""Problem configuration for optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from gemseo import execute_algo
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

from gemseo_benchmark.benchmarker.optimization_worker import OptimizationWorker
from gemseo_benchmark.data_profiles.targets_generator import TargetsGenerator
from gemseo_benchmark.problems.base_problem_configuration import (
    BaseProblemConfiguration,
)
from gemseo_benchmark.report.axis_data import IterationData
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from gemseo.algos.doe.base_doe_library import DriverLibraryOptionType
    from gemseo.algos.optimization_problem import OptimizationProblem

    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.data_profiles.target_values import TargetValues
    from gemseo_benchmark.problems.base_problem_configuration import (
        InputStartingPointsType,
    )


class OptimizationProblemConfiguration(BaseProblemConfiguration):
    """Problem configuration for optimization.

    An *optimization* problem configuration is a problem of reference
    to be solved by optimization algorithms for comparison purposes.
    An optimization problem configuration is characterized by
    its objective and constraint functions,
    its starting points (the optimization trajectories will start from each of them),
    and its target values (refer to :mod:`.data_profiles.target_values`).
    """

    abscissa_data_type: Final[type[IterationData]] = IterationData

    __optimization_problem: OptimizationProblem
    """A problem of the configuration."""

    performance_measure_label: ClassVar[str] = "Best feasible objective value"

    __targets_generator: TargetsGenerator | None
    """The generator of target values for the optimization problem configuration."""

    worker: ClassVar[type[OptimizationWorker]] = OptimizationWorker

    def __init__(
        self,
        name: str,
        create_problem: Callable[[], OptimizationProblem],
        starting_points: InputStartingPointsType = (),
        target_values: TargetValues | None = None,
        doe_algo_name: str = "",
        doe_size: int | None = None,
        doe_options: Mapping[str, DriverLibraryOptionType] = READ_ONLY_EMPTY_DICT,
        description: str = "No description available.",
        target_values_algorithms_configurations: AlgorithmsConfigurations | None = None,
        target_values_number: int | None = None,
        optimum: float | None = None,
    ) -> None:
        """
        Args:
            target_values_algorithms_configurations: The configurations of the
                optimization algorithms for the computation of target values.
                If ``None``, the target values will not be computed.
            target_values_number: The number of target values to compute.
                If ``None``, the target values will not be computed.
                N.B. the number of target values shall be the same for all the
                problem configurations of a same group.
        """  # noqa: D205, D212, D415
        self.__optimization_problem = create_problem()
        super().__init__(
            name,
            create_problem,
            target_values,
            starting_points,
            self.__optimization_problem.design_space,
            doe_algo_name,
            doe_size,
            doe_options,
            description,
            optimum,
            len(self.__optimization_problem.scalar_constraint_names),
        )
        self.__targets_generator = None
        if (
            target_values is None
            and target_values_algorithms_configurations is not None
            and target_values_number is not None
        ):
            self.compute_target_values(
                target_values_number, target_values_algorithms_configurations
            )

    @property
    def targets_generator(self) -> TargetsGenerator:
        """The generator for target values."""
        return self.__targets_generator

    # TODO: Remove after refactoring the Benchmarker.
    def __iter__(self) -> OptimizationProblem:
        """Iterate on the problem instances with respect to the starting points."""
        for starting_point in self.starting_points:
            problem = self.create_problem()
            problem.design_space.set_current_value(starting_point)
            yield problem

    @property
    def objective_name(self) -> str:
        """The name of the objective function."""
        return self.__optimization_problem.objective.name

    @property
    def constraints_names(self) -> list[str]:
        """The names of the scalar constraints."""
        return self.__optimization_problem.scalar_constraint_names

    def compute_target_values(
        self,
        targets_number: int,
        algorithm_configurations: AlgorithmsConfigurations,
        only_feasible: bool = True,
        budget_min: int = 1,
        show: bool = False,
        file_path: str = "",
        best_target_tolerance: float = 0.0,
    ) -> TargetValues:
        """Compute target values based on algorithm configurations.

        Args:
            targets_number: The number of target values to generate.
            algorithm_configurations: The algorithm configurations.
            only_feasible: Whether to generate only feasible target values.
            budget_min: The evaluation budget to be used to define
                the easiest target value.
            show: Whether to show the plot.
            file_path: The path where to save the plot.
                If empty, the plot is not saved.
            best_target_tolerance: The relative tolerance for comparisons with the
                best target value.

        Returns:
            The generated target values.
        """
        self.__targets_generator = TargetsGenerator()
        for configuration in algorithm_configurations:
            for instance in self:
                execute_algo(
                    instance,
                    "opt",
                    algo_name=configuration.algorithm_name,
                    **configuration.algorithm_options,
                )
                history = PerformanceHistory.from_problem(instance)
                self.__targets_generator.add_history(history=history)

        target_values = self.__targets_generator.compute_target_values(
            targets_number,
            budget_min,
            only_feasible,
            show,
            file_path,
            self.optimum,
            best_target_tolerance,
        )
        if not self.minimize_performance_measure:
            target_values.switch_performance_measure_sign()

        self.target_values = target_values
        return self.target_values

    # TODO: Use this method in Report.
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
    def minimize_performance_measure(self) -> bool:
        """Whether the performance measure is to be minimized."""
        return self.__optimization_problem.minimize_objective
