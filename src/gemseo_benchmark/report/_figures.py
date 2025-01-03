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

"""The figures dedicated to a group of benchmarking problems."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

import matplotlib.pyplot
import numpy
from gemseo.utils.matplotlib_figure import save_show_figure
from matplotlib.ticker import MaxNLocator

from gemseo_benchmark import _get_configuration_plot_options
from gemseo_benchmark import join_substrings
from gemseo_benchmark.results.performance_histories import PerformanceHistories
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from gemseo_benchmark import ConfigurationPlotOptions
    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )
    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.problems.problem import Problem
    from gemseo_benchmark.problems.problems_group import ProblemsGroup
    from gemseo_benchmark.results.results import Results


class Figures:
    """The figures dedicated to a group of benchmarking problems."""

    __algorithm_configurations: AlgorithmsConfigurations
    """The algorithm configurations."""

    __ALPHA: Final[float] = 0.3
    """The opacity level for overlapping areas.
    (Refer to the Matplotlib documentation.)"""

    __destination_dir: Path
    """The path to the root directory for the figures."""

    __GRID_KWARGS: Final[Mapping[str, str]] = {"visible": True, "linestyle": ":"}
    """The keyword arguments of `matplotlib.pyplot.grid`."""

    __group: ProblemsGroup
    """The group of problems to be represented."""

    __infeasibility_tolerance: int | float
    """The tolerance on the infeasibility measure."""

    __max_eval_number: int
    """The maximum number of evaluations to be displayed on the figures."""

    __plot_kwargs: Mapping[str, ConfigurationPlotOptions]
    """The keyword arguments of `matplotlib.axes.Axes.plot`
      for each algorithm configuration."""

    __results: Results
    """The paths to the reference histories
    for each algorithm configuration and reference problem."""

    __TARGET_VALUES_PLOT_KWARGS: ClassVar[Mapping[str, str | float]] = {
        "color": "red",
        "linestyle": ":",
        "zorder": 1.9,
    }
    """The keyword arguments for `matplotlib.axes.Axes.axhline`
    when plotting target values."""

    class _FileName(enum.Enum):
        """The name of a figure file."""

        DATA_PROFILE = "data_profile.png"
        PERFORMANCE_MEASURE = "performance_measure.png"
        PERFORMANCE_MEASURE_FOCUS = "performance_measure_focus.png"

    def __init__(
        self,
        algorithm_configurations: AlgorithmsConfigurations,
        group: ProblemsGroup,
        results: Results,
        destination_dir: Path,
        infeasibility_tolerance: float,
        max_eval_number: int,
        plot_kwargs: Mapping[str, ConfigurationPlotOptions],
    ) -> None:
        """
        Args:
            algorithm_configurations: The algorithm configurations.
            group: The group of problems.
            results: The paths to the reference histories
                for each algorithm configuration and reference problem.
            destination_dir: The path to the root directory for the figures.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum number of evaluations to be displayed
                on the figures.
                If 0, all the evaluations are displayed.
            plot_kwargs: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.
        """  # noqa: D205, D212, D415
        self.__algorithm_configurations = algorithm_configurations
        self.__destination_dir = destination_dir
        self.__group = group
        self.__infeasibility_tolerance = infeasibility_tolerance
        self.__max_eval_number = max_eval_number
        self.__plot_kwargs = _get_configuration_plot_options(
            plot_kwargs, algorithm_configurations.names
        )
        self.__results = results

    def plot_data_profiles(self) -> Path:
        """Plot the data profiles of the group of problems.

        Returns:
            The path to the figure.
        """
        plot_path = self.__get_data_profiles_path()
        self.__group.compute_data_profile(
            self.__algorithm_configurations,
            self.__results,
            show=False,
            plot_path=plot_path,
            infeasibility_tolerance=self.__infeasibility_tolerance,
            max_eval_number=self.__max_eval_number,
            plot_kwargs=self.__plot_kwargs,
            grid_kwargs=self.__GRID_KWARGS,
        )
        return plot_path

    def __get_data_profiles_path(self) -> Path:
        """Return the path to the data profiles of the group of problems."""
        return self.__destination_dir / self._FileName.DATA_PROFILE.value

    def plot(
        self,
        plot_all_histories: bool,
        use_performance_log_scale: bool,
        plot_only_median: bool,
    ) -> dict[str, dict[str, str]]:
        """Plot the figures for each problem of the group.

        Args:
            plot_all_histories: Whether to plot all the performance histories.
            use_performance_log_scale: Whether to use a logarithmic scale
                for the performance measure axis.
            plot_only_median: Whether to plot only the median and no other centile.

        Returns:
            The paths to the figures.
            The keys are the names of the problems and the values
            are the corresponding dictionaries of figures.
        """
        problems_to_figures = {}
        for problem in self.__group:
            problem_dir = self.__destination_dir / join_substrings(problem.name)
            problem_dir.mkdir()
            # Gather the performance histories
            performance_histories = {
                algorithm_configuration: PerformanceHistories(*[
                    PerformanceHistory.from_file(path)
                    for path in self.__results.get_paths(
                        algorithm_configuration.name, problem.name
                    )
                ])
                .cumulate_minimum()
                .get_equal_size_histories()
                for algorithm_configuration in self.__algorithm_configurations
            }
            # Draw the plots dedicated to each problem
            problems_to_figures[problem.name] = self.__get_problem_figures(
                problem,
                performance_histories,
                problem_dir,
                plot_all_histories,
                use_performance_log_scale,
                plot_only_median,
            )

        return problems_to_figures

    def __get_problem_figures(
        self,
        problem: Problem,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        destination_dir: Path,
        plot_all_histories: bool,
        use_performance_log_scale: bool,
        plot_only_median: bool,
    ) -> dict[str, str]:
        """Return the results figures of a problem.

        Args:
            problem: The problem.
            performance_histories: The performance histories for the problem.
            destination_dir: The path to the root directory for the figures.
            plot_all_histories: Whether to plot all the performance histories.
            use_performance_log_scale: Whether to use a logarithmic scale
                for the performance measure axis.
            plot_only_median: Whether to plot only the median and no other centile.

        Returns:
            The paths to the figures.
        """
        performance_path, focus_path = self.__plot_performance_measure(
            problem,
            performance_histories,
            destination_dir,
            plot_all_histories,
            use_performance_log_scale,
            plot_only_median,
        )
        return {
            self._FileName.DATA_PROFILE.value: self.__plot_data_profiles(
                problem, destination_dir
            ),
            self._FileName.PERFORMANCE_MEASURE.value: performance_path,
            self._FileName.PERFORMANCE_MEASURE_FOCUS.value: focus_path,
        }

    def __plot_data_profiles(self, problem: Problem, destination_dir: Path) -> Path:
        """Plot the data profiles for a problem.

        Args:
            problem: The problem.
            destination_dir: The destination directory for the figure.

        Returns:
            The path to the figure.
        """
        if len(self.__group) == 1:
            # Return the path to the data profiles of the group.
            path = self.__get_data_profiles_path()
            if path.is_file():
                return path

            return self.plot_data_profiles()

        path = destination_dir / self._FileName.DATA_PROFILE.value
        problem.compute_data_profile(
            self.__algorithm_configurations,
            self.__results,
            False,
            path,
            self.__infeasibility_tolerance,
            self.__max_eval_number,
            self.__plot_kwargs,
            self.__GRID_KWARGS,
        )
        return path

    def __plot_performance_measure(
        self,
        problem: Problem,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        destination_dir: Path,
        plot_all_histories: bool,
        use_performance_log_scale: bool,
        plot_only_median: bool,
    ) -> tuple[Path, Path]:
        """Plot the range of the performance measure for a problem.

        Args:
            problem: The problem.
            performance_histories: The performance histories for the problem.
            destination_dir: The path to the root directory for the figures.
            plot_all_histories: Whether to plot all the performance histories.
            use_performance_log_scale: Whether to use a logarithmic scale
                for the performance measure axis.
            plot_only_median: Whether to plot only the median and no other centile.

        Returns:
            The path to the main figure
            and the path to a focus on the target values.
        """
        figure = matplotlib.pyplot.figure()
        axes = figure.gca()
        if plot_all_histories:
            for algorithm_configuration, histories in performance_histories.items():
                name = algorithm_configuration.name
                for history in histories:
                    history.plot(axes, False, color=self.__plot_kwargs[name]["color"])
        # Find the maximum feasible objective value
        max_feasible_objective = max(
            history.remove_leading_infeasible()[0].objective_value
            for histories in performance_histories.values()
            for history in histories
            if history[-1].is_feasible
        )
        self.__plot_range(
            axes,
            performance_histories,
            lambda history: [
                item.objective_value if item.is_feasible else numpy.nan
                for item in history
            ],
            "Performance measure",
            max_feasible_objective,
            plot_only_median,
        )
        problem.target_values.plot_on_axes(
            axes,
            self.__TARGET_VALUES_PLOT_KWARGS,
            set_ylabel_kwargs={"rotation": 270, "labelpad": 20},
        )
        axes.grid(**self.__GRID_KWARGS)
        if use_performance_log_scale:
            axes.set_yscale("log")

        path = destination_dir / self._FileName.PERFORMANCE_MEASURE.value
        save_show_figure(figure, False, path)

        # Plot a focus on the target values
        performance_axes, targets_axes = figure.axes
        performance_axes.set_ylim(
            bottom=min(
                [
                    history[-1]
                    for histories in performance_histories.values()
                    for history in histories
                ]
                + list(problem.target_values)
            ).objective_value,
            top=max(
                target for target in problem.target_values if target.is_feasible
            ).objective_value,
        )
        targets_axes.set_ylim(performance_axes.get_ylim())
        focus_path = destination_dir / self._FileName.PERFORMANCE_MEASURE_FOCUS.value
        save_show_figure(figure, False, focus_path)
        return path, focus_path

    def __plot_range(
        self,
        axes: matplotlib.axes.Axes,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        data_getter: callable[[PerformanceHistory], float],
        y_label: str,
        max_feasible_objective: float,
        plot_only_median: bool,
    ) -> None:
        """Plot the range of data from performance histories.

        Args:
            axes: The axes of the plot.
            performance_histories: The performance histories
                of each algorithm configuration.
            data_getter: A function that gets the data of interest
                from a performance history.
            y_label: The label of the vertical axis.
            max_feasible_objective: The maximum feasible value
                of the objective function.
            plot_only_median: Whether to plot only the median and no other centile.
        """
        for algo_config, histories in performance_histories.items():
            name = algo_config.name
            data = numpy.array([
                data_getter(history) for history in histories.get_equal_size_histories()
            ])
            if not plot_only_median:
                PerformanceHistories.plot_centiles_range(
                    data,
                    axes,
                    (0, 100),
                    {"alpha": self.__ALPHA, "color": self.__plot_kwargs[name]["color"]},
                    max_feasible_objective,
                )

            PerformanceHistories.plot_median(data, axes, self.__plot_kwargs[name])

        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes.grid(**self.__GRID_KWARGS)
        axes.set_xlabel("Number of functions evaluations")
        axes.set_ylabel(y_label)
        axes.legend()
