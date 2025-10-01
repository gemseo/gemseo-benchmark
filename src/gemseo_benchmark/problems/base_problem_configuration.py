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
"""The interface for problem configurations."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any

from gemseo import compute_doe
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from numpy import array
from numpy import load
from numpy import ndarray
from numpy import save

from gemseo_benchmark.data_profiles.data_profile import DataProfile
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from pathlib import Path

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.base_doe_library import DriverLibraryOptionType
    from gemseo.typing import RealArray

    from gemseo_benchmark import ConfigurationPlotOptions
    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.benchmarker.base_worker import BaseWorker
    from gemseo_benchmark.data_profiles.target_values import TargetValues
    from gemseo_benchmark.report.axis_data import AbscissaData
    from gemseo_benchmark.results.results import Results

InputStartingPointsType = ndarray | Iterable[ndarray]


class BaseProblemConfiguration(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for problem configurations.

    A *problem configuration* is a problem of reference
    to be solved by iterative algorithms for comparison purposes.
    A problem configuration is characterized by the function that evaluates
    its *performance measure*
    (ex: the objective function of an optimization problem,
    or the residual for a multidisciplinary analysis),
    its *starting points*
    (an algorithm trajectory will start from each of them),
    and its *target values*
    (refer to the [target_values module][gemseo_benchmark.data_profiles.target_values]).
    """

    __create_problem: Callable[[], Any]
    """The function to create a problem of the configuration.
    (ex:
    an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem],
    a [BaseMDA][gemseo.mda.base_mda])."""

    __description: str
    """The description of the problem configuration."""

    __minimization_target_values: TargetValues | None
    """The target values for minimization."""

    __name: str
    """The name of the problem configuration."""

    __optimum: float | None
    """"The best performance measure known for the problem configuration."""

    __starting_points: list[RealArray]
    """The starting points to pass to the algorithm configurations."""

    __target_values: TargetValues | None
    """"The target values to compute data profiles."""

    __variable_space: DesignSpace | None
    """The space of the variables of the problem configuration."""

    def __init__(
        self,
        name: str,
        create_problem: Callable[[], Any],
        target_values: TargetValues | None,
        starting_points: InputStartingPointsType,
        variable_space: DesignSpace,
        doe_algo_name: str,
        doe_size: int | None,
        doe_options: Mapping[str, Any],
        description: str,
        optimum: float | None,
        number_of_scalar_constraints: int,
    ) -> None:
        """
        Args:
            name: The name of the problem configuration.
            create_problem: A function to create a problem of the configuration.
                !!! warning
                    If multiprocessing is intended when executing
                    algorithm configurations on the problem,
                    ``create_problem`` has to be pickable.
            target_values: The target values of the problem configuration.
            starting_points: The starting points of the problem configuration.
                If empty:
                if ``doe_algo_name`` is not empty
                then the starting points will be generated as a DOE;
                otherwise the current value of the optimization problem
                will be set as the single starting point.
            variable_space: The space of the problem variables.
            doe_algo_name: The name of the DOE algorithm.
                If empty and ``starting_points`` is empty,
                the current point of the variable space
                is set as the only starting point.
            doe_size: The number of starting points.
                If ``None``,
                this number is set as the problem dimension or 10 if bigger.
            doe_options: The options of the DOE algorithm.
            description: The description of the problem configuration
                (to appear in a benchmarking report).
            optimum: The best feasible performance measure of the problem configuration.
                If ``None``, it will not be set.
            number_of_scalar_constraints: The number of scalar constraints.
        """  # noqa: D205, D212, D415
        self.__create_problem = create_problem
        self.__description = description
        self.__minimization_target_values = None
        self.__name = name
        self.__number_of_scalar_constraints = number_of_scalar_constraints
        self.__optimum = optimum
        self.__starting_points = []
        self.__target_values = None
        self.__variable_space = variable_space

        if len(starting_points) > 0:
            self.starting_points = starting_points
        elif doe_algo_name:
            self.starting_points = self.__get_starting_points(
                doe_algo_name, doe_size, doe_options
            )
        else:
            default_starting_point = self._get_default_starting_point()
            if default_starting_point is None:
                self.starting_points = []
            else:
                self.starting_points = [default_starting_point]

        if target_values is not None:
            self.target_values = target_values

    @property
    def create_problem(self) -> Callable[[], Any]:
        """The function to create a problem of the configuration.

        The return type of this function depends on the type of the underlying
        |g| object (ex:
        [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem],
        [BaseMDA][gemseo.mda.base_mda]
        ).
        """
        return self.__create_problem

    @property
    def description(self) -> str:
        """The description of the problem configuration."""
        return self.__description

    @property
    def name(self) -> str:
        """The name of the problem configuration."""
        return self.__name

    @property
    def optimum(self) -> float:
        """The best feasible performance measure known for the problem configuration."""
        return self.__optimum

    @property
    def starting_points(self) -> list[ndarray]:
        """The starting points for the algorithm configurations.

        Raises:
            TypeError: If the starting points are neither passed as a NumPy array
                nor as an iterable.
            ValueError: If the problem has no starting point,
                or if the starting points are passed as a NumPy array that is not
                2-dimensional,
                or if one of the starting points does not have the same dimension
                as the problem.
        """
        if not self.__starting_points:
            msg = "The problem configuration has no starting point."
            raise ValueError(msg)

        return self.__starting_points

    @starting_points.setter
    def starting_points(self, starting_points: InputStartingPointsType) -> None:
        message = (
            "The starting points shall be passed as (lines of) a 2-dimensional "
            "NumPy array, or as an iterable of 1-dimensional NumPy arrays."
        )
        if isinstance(starting_points, ndarray):
            if starting_points.ndim != 2:
                msg = (
                    f"{message} A {starting_points.ndim}-dimensional NumPy array "
                    "was passed."
                )
                raise ValueError(msg)

            if starting_points.shape[1] != self.__variable_space.dimension:
                msg = (
                    f"{message} The number of columns ({starting_points.shape[1]}) "
                    f"is different from the problem dimension "
                    f"({self.__variable_space.dimension})."
                )
                raise ValueError(msg)

            starting_points_list = list(starting_points)
        else:
            try:
                iter(starting_points)
            except TypeError:
                msg = (
                    f"{message} The following type was passed: {type(starting_points)}."
                )
                raise TypeError(msg) from None

            self.__check_iterable_starting_points(starting_points)
            starting_points_list = list(starting_points)

        for point in starting_points_list:
            self.__variable_space.check_membership(point)

        self.__starting_points = starting_points_list

    def __check_iterable_starting_points(
        self, starting_points: Iterable[ndarray]
    ) -> None:
        """Check starting points passed as an iterable.

        Args:
            starting_points: The starting points.

        Raises:
            ValueError: If the iterable contains NumPy arrays of the wrong shape.
        """
        if any(
            point.ndim != 1 or point.size != self.dimension for point in starting_points
        ):
            msg = (
                "A starting point must be a 1-dimensional NumPy array of size "
                f"{self.dimension}."
            )
            raise ValueError(msg)

    def __get_starting_points(
        self,
        doe_algo_name: str,
        doe_size: int | None,
        doe_options: Mapping[str, DriverLibraryOptionType],
    ) -> ndarray:
        """Return the starting points of the problem configuration.

        Args:
            doe_algo_name: The name of the DOE algorithm.
            doe_size: The number of starting points.
                If ``None``, this number is set as the problem dimension or 10 if
                bigger.
            doe_options: The options of the DOE algorithm.

        Returns:
            The starting points.
        """
        if doe_size is None:
            doe_size = min([self.dimension, 10])

        return compute_doe(
            self.__variable_space,
            algo_name=doe_algo_name,
            n_samples=doe_size,
            **doe_options,
        )

    def _get_default_starting_point(self) -> RealArray | None:
        """Return the default starting point of the problem configuration.

        Return:
            The current value of the design space if it has one,
            ``None`` otherwise.
        """
        if self.__variable_space.has_current_value:
            return self.__variable_space.get_current_value()

        return None

    @property
    def target_values(self) -> TargetValues:
        """The target values of the problem configuration.

        Raises:
            ValueError: If the problem configuration has no target value.
        """
        if self.__target_values is None:
            msg = "The problem configuration has no target value."
            raise ValueError(msg)

        return self.__target_values

    @target_values.setter
    def target_values(self, target_values: TargetValues) -> None:
        self.__target_values = target_values
        self.__set_minimization_target_values()

    def save_starting_points(self, path: Path) -> None:
        """Save the starting points as a NumPy binary.

        Args:
            path: The path to the NumPy binary.
        """
        save(path, array(self.starting_points))

    def load_starting_point(self, path: Path) -> None:
        """Load the starting points from a NumPy binary.

        Args:
            path: The path to the NumPy binary.
        """
        self.starting_points = load(path)

    @property
    def dimension(self) -> int:
        """The dimension of the problem configuration."""
        return self.__variable_space.dimension

    def compute_data_profile(
        self,
        algos_configurations: AlgorithmsConfigurations,
        results: Results,
        show: bool = False,
        file_path: str | Path = "",
        infeasibility_tolerance: float = 0.0,
        max_iteration_number: int = 0,
        plot_settings: Mapping[str, ConfigurationPlotOptions] = READ_ONLY_EMPTY_DICT,
        grid_settings: Mapping[str, str] = READ_ONLY_EMPTY_DICT,
        use_abscissa_log_scale: bool = False,
    ) -> None:
        """Compute the data profiles of given algorithms.

        Args:
            algos_configurations: The algorithms configurations.
            results: The paths to the reference histories for each algorithm.
            show: Whether to display the plot.
            file_path: The path where to save the plot.
                If empty, the plot is not saved.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_iteration_number: The maximum number of iterations to plot.
                If ``0``, this value is inferred from the longest history.
            plot_settings: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.
            grid_settings: The keyword arguments of `matplotlib.pyplot.grid`.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
        """
        data_profile = DataProfile({self.name: self.minimization_target_values})
        for configuration_name in algos_configurations.names:
            for history_path in results.get_paths(configuration_name, self.name):
                history = PerformanceHistory.from_file(history_path)
                if max_iteration_number:
                    history = history.shorten(max_iteration_number)

                history.apply_infeasibility_tolerance(infeasibility_tolerance)
                data_profile.add_history(
                    self.name,
                    configuration_name,
                    history.performance_measures,
                    history.infeasibility_measures,
                )

        data_profile.plot(
            show=show,
            file_path=file_path,
            plot_settings=plot_settings,
            grid_settings=grid_settings,
            use_abscissa_log_scale=use_abscissa_log_scale,
        )

    @property
    def minimization_target_values(self) -> TargetValues:
        """The target values for the minimization of the performance measure."""
        return self.__minimization_target_values

    @property
    @abstractmethod
    def minimize_performance_measure(self) -> bool:
        """Whether the performance measure of the problem is to be minimized."""

    def __set_minimization_target_values(self) -> None:
        """Set the target values for the minimization of the performance measure."""
        if self.minimize_performance_measure:
            self.__minimization_target_values = self.__target_values
        else:
            self.__minimization_target_values = deepcopy(self.__target_values)
            self.__minimization_target_values.switch_performance_measure_sign()

    @property
    def variable_space(self) -> DesignSpace:
        """The space of the problem variables."""
        return self.__variable_space

    @property
    @abstractmethod
    def worker(self) -> type[BaseWorker]:
        """The type of benchmarking worker."""

    @property
    def number_of_scalar_constraints(self) -> int:
        """The number of scalar constraints."""
        return self.__number_of_scalar_constraints

    @property
    @abstractmethod
    def performance_measure_label(self) -> str:
        """The label for the performance measure axis."""

    @property
    @abstractmethod
    def abscissa_data_type(self) -> type[AbscissaData]:
        """The type of abscissa axis data."""
