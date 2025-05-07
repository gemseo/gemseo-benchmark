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

import datetime
import enum
import functools
import logging
from math import isinf
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import Union

import matplotlib.pyplot
import numpy
import pandas
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.string_tools import pretty_str
from matplotlib.ticker import MaxNLocator

from gemseo_benchmark import _get_configuration_plot_options
from gemseo_benchmark import join_substrings
from gemseo_benchmark.results.performance_histories import PerformanceHistories
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo_benchmark import ConfigurationPlotOptions
    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )
    from gemseo_benchmark.algorithms.algorithms_configurations import (
        AlgorithmsConfigurations,
    )
    from gemseo_benchmark.problems.base_benchmarking_problem import (
        BaseBenchmarkingProblem,
    )
    from gemseo_benchmark.problems.problems_group import ProblemsGroup
    from gemseo_benchmark.results.history_item import HistoryItem
    from gemseo_benchmark.results.results import Results

LOGGER = logging.getLogger(__name__)


class Figures:
    """The figures dedicated to a group of benchmarking problems."""

    __algorithm_configurations: AlgorithmsConfigurations
    """The algorithm configurations."""

    __ALPHA: Final[float] = 0.3
    """The opacity level for overlapping areas.
    (Refer to the Matplotlib documentation.)"""

    __directory_path: Path
    """The path to the root directory for the figures."""

    __GRID_KWARGS: Final[Mapping[str, str]] = {"visible": True, "linestyle": ":"}
    """The keyword arguments of `matplotlib.pyplot.grid`."""

    __group: ProblemsGroup
    """The group of problems to be represented."""

    __infeasibility_tolerance: int | float
    """The tolerance on the infeasibility measure."""

    __max_eval_number: int
    """The maximum number of evaluations to be displayed on the figures."""

    __MATPLOTLIB_LOG_SCALE: Final[str] = "log"
    """The Matplotlib value for logarithmic scale."""

    __MATPLOTLIB_SYMMETRIC_LOG_SCALE: Final[str] = "symlog"
    """The Matplotlib value for symmetric logarithmic scale."""

    __plot_kwargs: Mapping[str, ConfigurationPlotOptions]
    """The keyword arguments of `matplotlib.axes.Axes.plot`
      for each algorithm configuration."""

    __results: Results
    """The paths to the reference histories
    for each algorithm configuration and reference problem."""

    __TABLE_PERCENTILES: Final[dict[str, int]] = {
        "maximum": 100,
        "75th centile": 75,
        "median": 50,
        "25th centile": 25,
        "minimum": 0,
    }
    """The percentiles to be displayed in the report tables."""

    __TARGET_VALUES_PLOT_KWARGS: ClassVar[Mapping[str, str | float]] = {
        "color": "red",
        "linestyle": ":",
        "zorder": 1.9,
    }
    """The keyword arguments for `matplotlib.axes.Axes.axhline`
    when plotting target values."""

    class _FigureFileName(enum.Enum):
        """The name of a figure file."""

        DATA_PROFILE = "data_profile.png"
        EXECUTION_TIME = "execution_time.png"
        INFEASIBILITY_MEASURE = "infeasibility_measure.png"
        NUMBER_OF_UNSATISFIED_CONSTRAINTS = "number_of_unsatisfied_constraints.png"
        PERFORMANCE_MEASURE = "performance_measure.png"
        PERFORMANCE_MEASURE_FOCUS = "performance_measure_focus.png"

    ProblemFigurePaths = dict[
        Union[_FigureFileName, str], Union[Path, dict[_FigureFileName, Path]]
    ]
    """The paths to the figures dedicated to a problem."""

    class _TableFileName(enum.Enum):
        """The name of a table file."""

        INFEASIBILITY_MEASURE = "infeasibility_measure.csv"
        NUMBER_OF_UNSATISFIED_CONSTRAINTS = "number_of_unsatisfied_constraints.csv"
        PERFORMANCE_MEASURE = "performance_measure.csv"

    ProblemTablePaths = dict[
        Union[_TableFileName, str], Union[Path, dict[_TableFileName, Path]]
    ]
    """The paths to the tables dedicated to a problem."""

    def __init__(
        self,
        algorithm_configurations: AlgorithmsConfigurations,
        group: ProblemsGroup,
        results: Results,
        directory_path: Path,
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
            directory_path: The path to the root directory for the figures.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum number of evaluations to be displayed
                on the figures.
                If 0, all the evaluations are displayed.
            plot_kwargs: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.
        """  # noqa: D205, D212, D415
        self.__algorithm_configurations = algorithm_configurations
        self.__directory_path = directory_path
        self.__group = group
        self.__infeasibility_tolerance = infeasibility_tolerance
        self.__max_eval_number = max_eval_number
        self.__plot_kwargs = _get_configuration_plot_options(
            plot_kwargs, algorithm_configurations.names
        )
        self.__results = results

    def plot_data_profiles(self, use_evaluation_log_scale: bool = False) -> Path:
        """Plot the data profiles of the group of problems.

        Args:
            use_evaluation_log_scale: Whether to use a logarithmic scale
                for the number of function evaluations axis.

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
            use_evaluation_log_scale=use_evaluation_log_scale,
        )
        return plot_path

    def __get_data_profiles_path(self) -> Path:
        """Return the path to the data profiles of the group of problems."""
        return self.__directory_path / self._FigureFileName.DATA_PROFILE.value

    def plot(
        self,
        plot_all_histories: bool,
        use_performance_log_scale: bool,
        plot_only_median: bool,
        use_time_log_scale: bool,
        use_evaluation_log_scale: bool,
        table_values_format: str = ".6g",
        bypass_unequal_representation: bool = False,
    ) -> tuple[dict[str, ProblemFigurePaths], dict[str, ProblemTablePaths]]:
        """Plot the figures for each problem of the group.

        Args:
            plot_all_histories: Whether to plot all the performance histories.
            use_performance_log_scale: Whether to use a logarithmic scale
                for the performance measure axis.
            plot_only_median: Whether to plot only the median and no other centile.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.
            use_evaluation_log_scale: Whether to use a logarithmic scale
                for the number of function evaluations axis.
            table_values_format: The string format for the table values.
            bypass_unequal_representation: Whether to bypass the check that ensures
                that each algorithm configuration is represented by the same number of
                performance histories for a given problem.

        Returns:
            The paths to the figures and the paths to the tables.
            The keys are the names of the problems and the values
            are the corresponding dictionaries of figures or tables.

        Raises:
            ValueError: If ``bypass_unequal_representation`` is ``False`` and at least
                one algorithm configuration is represented by fewer performance
                histories than another on an given problem.
        """
        problems_to_figures = {}
        problems_to_tables = {}
        for problem in self.__group:
            problem_dir = self.__directory_path / join_substrings(problem.name)
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

            # Check the algorithm configurations are equally represented on the problem.
            number_of_histories = {
                algorithm_configuration.name: len(histories)
                for algorithm_configuration, histories in performance_histories.items()
            }
            if len(set(number_of_histories.values())) > 1:
                numbers = pretty_str(
                    [
                        f"{n_histories} for '{name}'"
                        for name, n_histories in number_of_histories.items()
                    ],
                    sort=False,
                    use_and=True,
                )
                message = (
                    "The number of performance histories varies for "
                    f"'{problem.name}': {numbers}."
                )
                if bypass_unequal_representation:
                    LOGGER.warning(message)
                else:
                    raise ValueError(message)

            max_feasible_performance = -float("inf")
            for histories in performance_histories.values():
                for history in histories:
                    feasible_history = history.remove_leading_infeasible()
                    if len(feasible_history) > 0:
                        max_feasible_performance = max(
                            feasible_history[0].objective_value,
                            max_feasible_performance,
                        )

            if not problem.minimize_performance_measure:
                max_feasible_performance = -max_feasible_performance
                for histories in performance_histories.values():
                    histories.switch_performance_measure_sign()

            # Draw the plots dedicated to each problem.
            problems_to_figures[problem.name] = self.__get_problem_figures(
                problem,
                performance_histories,
                problem_dir,
                plot_all_histories,
                use_performance_log_scale,
                plot_only_median,
                max_feasible_performance,
                use_time_log_scale,
                use_evaluation_log_scale,
            )
            # Fill the tables dedicated to each problem.
            problems_to_tables[problem.name] = self.__get_problem_tables(
                performance_histories,
                problem_dir,
                table_values_format,
                problem.minimize_performance_measure,
            )

        return problems_to_figures, problems_to_tables

    def __get_problem_figures(
        self,
        problem: BaseBenchmarkingProblem,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        plot_all_histories: bool,
        use_performance_log_scale: bool,
        plot_only_median: bool,
        max_feasible_performance: float,
        use_time_log_scale: bool,
        use_evaluation_log_scale: bool,
    ) -> ProblemFigurePaths:
        """Return the results figures of a problem.

        Args:
            problem: The problem.
            performance_histories: The performance histories for the problem.
            directory_path: The path to the root directory for the figures.
            plot_all_histories: Whether to plot all the performance histories.
            use_performance_log_scale: Whether to use a logarithmic scale
                for the performance measure axis.
            plot_only_median: Whether to plot only the median and no other centile.
            max_feasible_performance: The maximum feasible performance value.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.
            use_evaluation_log_scale: Whether to use a logarithmic scale
                for the number of function evaluations axis.

        Returns:
            The paths to the figures dedicated to the problem.
        """
        figures = {
            self._FigureFileName.DATA_PROFILE: self.__plot_data_profiles(
                problem, directory_path, use_evaluation_log_scale
            )
        }
        (
            figures[self._FigureFileName.PERFORMANCE_MEASURE],
            figures[self._FigureFileName.PERFORMANCE_MEASURE_FOCUS],
        ) = self.__plot_performance_measure(
            problem,
            performance_histories,
            directory_path,
            plot_all_histories,
            use_performance_log_scale,
            plot_only_median,
            use_evaluation_log_scale,
        )
        figures[self._FigureFileName.EXECUTION_TIME] = self.__plot_execution_time(
            performance_histories,
            directory_path,
            use_time_log_scale,
        )
        if problem.constraints_names:
            performance_measure_is_minimized = problem.minimize_performance_measure
            figures[self._FigureFileName.INFEASIBILITY_MEASURE] = (
                self.__plot_infeasibility_measure(
                    performance_histories,
                    directory_path,
                    plot_only_median,
                    use_evaluation_log_scale,
                    plot_all_histories,
                    performance_measure_is_minimized,
                )
            )
            figures[self._FigureFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS] = (
                self.__plot_number_of_unsatisfied_constraints(
                    performance_histories,
                    directory_path,
                    plot_only_median,
                    use_evaluation_log_scale,
                    plot_all_histories,
                    performance_measure_is_minimized,
                )
            )

        figures.update(
            self.__get_algorithms_plots(
                problem,
                performance_histories,
                max_feasible_performance,
                directory_path,
                use_evaluation_log_scale,
                use_performance_log_scale,
                plot_all_histories,
            )
        )
        return figures

    def __plot_data_profiles(
        self,
        problem: BaseBenchmarkingProblem,
        directory_path: Path,
        use_evaluation_log_scale: bool,
    ) -> Path:
        """Plot the data profiles for a problem.

        Args:
            problem: The problem.
            directory_path: The destination directory for the figure.
            use_evaluation_log_scale: Whether to use a logarithmic scale
                for the number of function evaluations axis.

        Returns:
            The path to the figure.
        """
        if len(self.__group) == 1:
            # Return the path to the data profiles of the group.
            file_path = self.__get_data_profiles_path()
            if file_path.is_file():
                return file_path

            return self.plot_data_profiles(use_evaluation_log_scale)

        file_path = directory_path / self._FigureFileName.DATA_PROFILE.value
        problem.compute_data_profile(
            self.__algorithm_configurations,
            self.__results,
            False,
            file_path,
            self.__infeasibility_tolerance,
            self.__max_eval_number,
            self.__plot_kwargs,
            self.__GRID_KWARGS,
            use_evaluation_log_scale,
        )
        return file_path

    def __plot_performance_measure(
        self,
        problem: BaseBenchmarkingProblem,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        plot_all_histories: bool,
        use_performance_log_scale: bool,
        plot_only_median: bool,
        use_evaluation_log_scale: bool,
    ) -> tuple[Path, Path]:
        """Plot the performance measure of algorithm configurations on a problem.

        Args:
            problem: The problem.
            performance_histories: The performance histories for the problem.
            directory_path: The path to the root directory for the figures.
            plot_all_histories: Whether to plot all the performance histories.
            use_performance_log_scale: Whether to use a logarithmic scale
                for the performance measure axis.
            plot_only_median: Whether to plot only the median and no other centile.
            use_evaluation_log_scale: Whether to use a logarithmic scale
                for the number of function evaluations axis.

        Returns:
            The path to the main figure
            and the path to a focus on the target values.
        """
        figure = matplotlib.pyplot.figure()
        axes = figure.gca()
        # Find the extremal feasible performance measure.
        worst_feasible_performances = [
            history.remove_leading_infeasible()[0].objective_value
            for histories in performance_histories.values()
            for history in histories
            if history[-1].is_feasible
        ]
        if problem.minimize_performance_measure:
            extremal_feasible_performance = max(worst_feasible_performances)
        else:
            extremal_feasible_performance = min(worst_feasible_performances)

        self.__plot_range(
            axes,
            performance_histories,
            functools.partial(
                self.__get_performance_measure,
                infeasible_performance_measure=float("nan"),
            ),
            "Performance measure",
            extremal_feasible_performance,
            plot_only_median,
            use_evaluation_log_scale,
            plot_all_histories,
            problem.minimize_performance_measure,
        )
        problem.target_values.plot_on_axes(
            axes,
            self.__TARGET_VALUES_PLOT_KWARGS,
            set_ylabel_kwargs={"rotation": 270, "labelpad": 20},
        )
        axes.grid(**self.__GRID_KWARGS)
        if use_performance_log_scale:
            axes.set_yscale(self.__MATPLOTLIB_SYMMETRIC_LOG_SCALE)

        file_path = directory_path / self._FigureFileName.PERFORMANCE_MEASURE.value
        save_show_figure(figure, False, file_path)

        # Plot a focus on the target values
        performance_axes, targets_axes = figure.axes
        history_items = [
            history[-1]
            for histories in performance_histories.values()
            for history in histories
        ] + list(problem.minimization_target_values)
        target_items = [
            target
            for target in problem.minimization_target_values
            if target.is_feasible
        ]
        best_performance_measure = min(history_items).objective_value
        worst_performance_measure = max(target_items).objective_value
        if problem.minimize_performance_measure:
            args = (best_performance_measure, worst_performance_measure)
        else:
            args = (-worst_performance_measure, -best_performance_measure)

        performance_axes.set_ylim(*args)
        targets_axes.set_ylim(performance_axes.get_ylim())
        focus_file_path = (
            directory_path / self._FigureFileName.PERFORMANCE_MEASURE_FOCUS.value
        )
        save_show_figure(figure, False, focus_file_path)
        return file_path, focus_file_path

    @staticmethod
    def __get_performance_measure(
        item: HistoryItem, infeasible_performance_measure: float
    ) -> float:
        """Return the performance measure of a history item.

        Args:
            item: The history item.
            infeasible_performance_measure: The performance measure to return
                for infeasible history items.

        Returns:
            The performance measure of the history item.
        """
        return (
            item.objective_value if item.is_feasible else infeasible_performance_measure
        )

    def __plot_infeasibility_measure(
        self,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        plot_only_median: bool,
        use_evaluation_log_scale: bool,
        plot_all_histories: bool,
        performance_measure_is_minimized: bool,
    ) -> Path:
        """Plot the infeasibility measure of algorithm configurations on a problem.

        Args:
            performance_histories: The performance histories for the problem.
            directory_path: The path to the root directory for the figures.
            plot_only_median: Whether to plot only the median and no other centile.
            use_evaluation_log_scale: Whether to use a logarithmic scale
                for the number of function evaluations axis.
            plot_all_histories: Whether to plot all the performance histories.
            performance_measure_is_minimized: Whether the performance measure
                is minimized (rather than maximized).

        Returns:
            The path to the figure.
        """
        file_path = directory_path / self._FigureFileName.INFEASIBILITY_MEASURE.value
        figure = matplotlib.pyplot.figure()
        axes = figure.gca()
        axes.set_yscale(self.__MATPLOTLIB_LOG_SCALE)
        self.__plot_range(
            axes,
            performance_histories,
            self.__get_infeasibility_measure,
            "Infeasibility measure",
            None,
            plot_only_median,
            use_evaluation_log_scale,
            plot_all_histories,
            performance_measure_is_minimized,
        )
        save_show_figure(figure, False, file_path)
        return file_path

    @staticmethod
    def __get_infeasibility_measure(item: HistoryItem) -> float:
        """Return the infeasibility measure of a history item."""
        return item.infeasibility_measure

    def __plot_number_of_unsatisfied_constraints(
        self,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        plot_only_median: bool,
        use_evaluation_log_scale: bool,
        plot_all_histories: bool,
        performance_measure_is_minimized: bool,
    ) -> Path:
        """Plot the number of constraints unsatisfied by algorithm configurations.

        Args:
            performance_histories: The performance histories
                of each algorithm configuration.
            directory_path: The path to the directory where to save the figure.
            plot_only_median: Whether to plot only the median and no other centile.
            use_evaluation_log_scale: Whether to use a logarithmic scale
                for the number of function evaluations axis.
            plot_all_histories: Whether to plot all the performance histories.
            performance_measure_is_minimized: Whether the performance measure
                is minimized (rather than maximized).

        Returns:
            The path to the figure.
        """
        file_path = (
            directory_path
            / self._FigureFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS.value
        )
        figure = matplotlib.pyplot.figure()
        axes = figure.gca()
        self.__plot_range(
            axes,
            performance_histories,
            self.__get_number_of_unsatisfied_constraints,
            "Number of unsatisfied constraints",
            None,
            plot_only_median,
            use_evaluation_log_scale,
            plot_all_histories,
            performance_measure_is_minimized,
        )
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        save_show_figure(figure, False, file_path)
        return file_path

    @staticmethod
    def __get_number_of_unsatisfied_constraints(item: HistoryItem) -> int | float:
        """Return the number of unsatisfied constraints of a history item."""
        number = item.n_unsatisfied_constraints
        return numpy.nan if number is None else number

    def __plot_range(
        self,
        axes: matplotlib.axes.Axes,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        data_getter: callable[[HistoryItem], int | float],
        y_label: str,
        max_feasible_objective: float,
        plot_only_median: bool,
        use_evaluation_log_scale: bool,
        plot_all_histories: bool,
        performance_measure_is_minimized: bool,
    ) -> None:
        """Plot the range of data from performance histories.

        Args:
            axes: The axes of the plot.
            performance_histories: The performance histories
                of each algorithm configuration.
            data_getter: A function that gets the data of interest
                from an item of a performance history.
            y_label: The label of the vertical axis.
            max_feasible_objective: The maximum feasible value
                of the objective function.
            plot_only_median: Whether to plot only the median and no other centile.
            use_evaluation_log_scale: Whether to use a logarithmic scale
                for the number of function evaluations axis.
            plot_all_histories: Whether to plot all the performance histories.
            performance_measure_is_minimized: Whether the performance measure
                is minimized (rather than maximized).
        """
        for algo_config, histories in performance_histories.items():
            name = algo_config.name
            data = numpy.array([
                [data_getter(item) for item in history]
                for history in histories.get_equal_size_histories()
            ])
            if plot_all_histories:
                axes.plot(
                    range(1, data.shape[1] + 1),
                    data.T,
                    color=self.__plot_kwargs[name]["color"],
                )

            if not plot_only_median:
                PerformanceHistories.plot_centiles_range(
                    data,
                    axes,
                    (0, 100),
                    {"alpha": self.__ALPHA, "color": self.__plot_kwargs[name]["color"]},
                    max_feasible_objective,
                    performance_measure_is_minimized,
                )

            PerformanceHistories.plot_median(
                data, axes, self.__plot_kwargs[name], performance_measure_is_minimized
            )

        if use_evaluation_log_scale:
            axes.set_xscale(self.__MATPLOTLIB_LOG_SCALE)
        else:
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))

        axes.grid(**self.__GRID_KWARGS)
        axes.set_xlabel("Number of functions evaluations")
        axes.set_ylabel(y_label)
        axes.legend()

    def __plot_execution_time(
        self,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        use_log_scale: bool,
    ) -> Path:
        """Plot the execution time of algorithm configurations on a problem.

        Args:
            performance_histories: The performance histories
                of each algorithm configuration.
            directory_path: The path to the directory where to save the figure.
            use_log_scale: Whether to use a logarithmic time scale.

        Returns:
            The path to the figure.
        """
        file_path = directory_path / self._FigureFileName.EXECUTION_TIME.value
        figure, axes = matplotlib.pyplot.subplots()
        data = [
            [
                float("inf") if history.total_time is None else history.total_time
                for history in histories
            ]
            for histories in performance_histories.values()
        ]
        labels = [algo_config.name for algo_config in performance_histories]
        artists = axes.boxplot(data, whis=(0, 100), patch_artist=True, labels=labels)
        for index, (box, median) in enumerate(
            zip(artists["boxes"], artists["medians"])
        ):
            name = labels[index]
            color = self.__plot_kwargs[name]["color"]
            box.set(
                edgecolor=color,
                facecolor=(*matplotlib.colors.to_rgb(color), self.__ALPHA),
            )
            median.set(
                color=color, linewidth=2, marker=self.__plot_kwargs[name]["marker"]
            )
            for whisker in artists["whiskers"][2 * index : 2 * (index + 1)]:
                whisker.set(color=color)

            for cap in artists["caps"][2 * index : 2 * (index + 1)]:
                cap.set(color=color)

        axes.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, _: str(datetime.timedelta(seconds=x))
            )
        )
        axes.grid(axis="y", **self.__GRID_KWARGS)
        axes.tick_params(axis="x", labelrotation=45)
        axes.set_ylabel(f"Execution time{' (in seconds)' if use_log_scale else ''}")
        if use_log_scale:
            axes.set_yscale(self.__MATPLOTLIB_LOG_SCALE)

        axes.legend(
            artists["medians"],
            [algo_config.name for algo_config in performance_histories],
        )
        save_show_figure(figure, False, file_path)
        return file_path

    def __get_algorithms_plots(
        self,
        problem: BaseBenchmarkingProblem,
        performance_histories: Mapping[str, PerformanceHistories],
        max_feasible_performance: float,
        directory_path: Path,
        use_evaluation_log_scale: bool,
        use_performance_log_scale: bool,
        plot_all_histories: bool,
    ) -> dict[str, dict[_FigureFileName, Path]]:
        """Return the figures associated with algorithm configurations for a problem.

        Args:
            problem: The problem.
            performance_histories: The performance histories for the problem.
            max_feasible_performance: The maximum feasible performance value.
            directory_path: The path to the directory where to save the figures.
            use_evaluation_log_scale: Whether to use a logarithmic scale
                for the number of function evaluations axis.
            use_performance_log_scale: Whether to use a logarithmic scale
                for the performance measure axis.
            plot_all_histories: Whether to plot all the performance histories.

        Returns:
            The paths to the figures for each algorithm configuration.
        """
        figures = {}
        # Plot the performance measure distribution for each configuration
        performance_figures = {}
        for configuration in self.__algorithm_configurations:
            figure, axes = matplotlib.pyplot.subplots()
            performance_histories[configuration].plot_performance_measure_distribution(
                axes,
                max_feasible_performance,
                plot_all_histories,
                problem.minimize_performance_measure,
            )
            if use_performance_log_scale:
                axes.set_yscale(self.__MATPLOTLIB_SYMMETRIC_LOG_SCALE)

            if use_evaluation_log_scale:
                axes.set_xscale(self.__MATPLOTLIB_LOG_SCALE)

            axes.grid(**self.__GRID_KWARGS)
            performance_figures[configuration] = figure

        self.__set_common_limits(performance_figures.values())

        for configuration, figure in performance_figures.items():
            # Add the target values and save the figure
            problem.target_values.plot_on_axes(
                figure.gca(), self.__TARGET_VALUES_PLOT_KWARGS
            )
            configuration_dir = directory_path / join_substrings(configuration.name)
            configuration_dir.mkdir()
            file_path = (
                configuration_dir / self._FigureFileName.PERFORMANCE_MEASURE.value
            )
            save_show_figure(figure, False, file_path)
            figures[configuration.name] = {
                self._FigureFileName.PERFORMANCE_MEASURE: file_path
            }
            # Focus on the targets qnd save another figure
            performance_axes, targets_axes = figure.axes
            performance_axes.autoscale(enable=True, axis="y", tight=True)
            if problem.minimize_performance_measure:
                performance_axes.set_ylim(
                    top=max(problem.target_values).objective_value
                )
            else:
                performance_axes.set_ylim(
                    bottom=min(problem.target_values).objective_value
                )

            targets_axes.set_ylim(performance_axes.get_ylim())
            file_path = (
                configuration_dir / self._FigureFileName.PERFORMANCE_MEASURE_FOCUS.value
            )
            save_show_figure(figure, False, file_path)
            figures[configuration.name][
                self._FigureFileName.PERFORMANCE_MEASURE_FOCUS
            ] = file_path

        if problem.constraints_names:
            # Plot the infeasibility measure distribution for each configuration
            infeasibility_figures = {}
            for configuration in self.__algorithm_configurations:
                figure, axes = matplotlib.pyplot.subplots()
                performance_histories[
                    configuration
                ].plot_infeasibility_measure_distribution(axes, plot_all_histories)
                axes.set_yscale(self.__MATPLOTLIB_LOG_SCALE)
                if use_evaluation_log_scale:
                    axes.set_xscale(self.__MATPLOTLIB_LOG_SCALE)

                axes.grid(**self.__GRID_KWARGS)
                infeasibility_figures[configuration] = figure

            self.__set_common_limits(infeasibility_figures.values())

            for configuration, figure in infeasibility_figures.items():
                file_path = (
                    directory_path
                    / join_substrings(configuration.name)
                    / self._FigureFileName.INFEASIBILITY_MEASURE.value
                )
                save_show_figure(figure, False, file_path)
                figures[configuration.name][
                    self._FigureFileName.INFEASIBILITY_MEASURE
                ] = file_path

            constraints_figures = {}
            for configuration in self.__algorithm_configurations:
                figure, axes = matplotlib.pyplot.subplots()
                performance_histories[
                    configuration
                ].plot_number_of_unsatisfied_constraints_distribution(
                    axes, plot_all_histories
                )
                axes.yaxis.set_major_locator(MaxNLocator(integer=True))  # TODO: move
                if use_evaluation_log_scale:
                    axes.set_xscale(self.__MATPLOTLIB_LOG_SCALE)

                axes.grid(**self.__GRID_KWARGS)
                constraints_figures[configuration] = figure

            self.__set_common_limits(constraints_figures.values())

            for configuration, figure in constraints_figures.items():
                file_path = (
                    directory_path
                    / join_substrings(configuration.name)
                    / self._FigureFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS.value
                )
                save_show_figure(figure, False, file_path)
                figures[configuration.name][
                    self._FigureFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS
                ] = file_path

        return figures

    def __set_common_limits(self, figures: Iterable[matplotlib.Figure]) -> None:
        """Set common limits to figures.

        Args:
            figures: The figures.
        """
        xlim = [float("inf"), -float("inf")]
        ylim = [float("inf"), -float("inf")]
        for figure in figures:
            for lim, get_lim in (
                (xlim, figure.gca().get_xlim),
                (ylim, figure.gca().get_ylim),
            ):
                lim_min, lim_max = get_lim()
                lim[0] = min(lim[0], lim_min)
                lim[1] = max(lim[1], lim_max)

        for figure in figures:
            figure.gca().set_xlim(*xlim)
            figure.gca().set_ylim(*ylim)

    @classmethod
    def __get_problem_tables(
        cls,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        table_values_format: str,
        performance_measure_is_minimized: bool,
    ) -> ProblemTablePaths:
        """Tabulate statistics on final data achieved by algorithm configurations.

        Args:
            performance_histories: The performance histories
                of each algorithm configuration.
            directory_path: The path to the directory where to save the CSV files.
            table_values_format: The string format for the table values.
            performance_measure_is_minimized: Whether the performance measure
                is minimized (rather than maximized).

        Returns:
            The paths to the tables dedicated to the problem.
        """
        tables = {configuration.name: {} for configuration in performance_histories}
        final_history_items = {
            configuration: [history[-1] for history in histories]
            for configuration, histories in performance_histories.items()
        }
        for data_getter, file_name in [
            (
                functools.partial(
                    cls.__get_performance_measure,
                    infeasible_performance_measure=float("inf")
                    if performance_measure_is_minimized
                    else -float("inf"),
                ),
                Figures._TableFileName.PERFORMANCE_MEASURE,
            ),
            (
                cls.__get_infeasibility_measure,
                Figures._TableFileName.INFEASIBILITY_MEASURE,
            ),
            (
                cls.__get_number_of_unsatisfied_constraints,
                Figures._TableFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS,
            ),
        ]:
            data = pandas.DataFrame(
                {
                    configuration.name: [
                        "infeasible"
                        if isinf(value)
                        else f"{{value:{table_values_format}}}".format(value=value)
                        for value in numpy.percentile(
                            [data_getter(item) for item in history_items],
                            tuple(cls.__TABLE_PERCENTILES.values()),
                            method="inverted_cdf",
                        )
                    ]
                    for configuration, history_items in final_history_items.items()
                },
                cls.__TABLE_PERCENTILES,
            )
            # Save the data for the whole group of algorithm configurations.
            file_path = directory_path / file_name.value
            data.iloc[::-1].T.to_csv(file_path)
            tables[file_name] = file_path
            # Save the data for each algorithm configuration.
            for configuration in performance_histories:
                file_path = (
                    directory_path
                    / join_substrings(configuration.name)
                    / file_name.value
                )
                data[configuration.name].to_csv(file_path)
                tables[configuration.name][file_name] = file_path

        return tables
