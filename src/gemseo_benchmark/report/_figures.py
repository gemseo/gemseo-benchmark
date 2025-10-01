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

"""The figures dedicated to a group of problem configurations."""

from __future__ import annotations

import enum
import functools
import logging
from math import isinf
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

import matplotlib.pyplot
import numpy
import pandas
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.string_tools import pretty_str
from matplotlib.ticker import MaxNLocator

from gemseo_benchmark import _get_configuration_plot_options
from gemseo_benchmark import join_substrings
from gemseo_benchmark.report.axis_data import ConstraintData
from gemseo_benchmark.report.axis_data import InfeasibilityData
from gemseo_benchmark.report.axis_data import PerformanceData
from gemseo_benchmark.report.axis_data import TimeAbscissaData
from gemseo_benchmark.report.axis_data import TimeOrdinateData
from gemseo_benchmark.report.range_plot import RangePlot
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
    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )
    from gemseo_benchmark.problems.problems_group import ProblemsGroup
    from gemseo_benchmark.results.history_item import HistoryItem
    from gemseo_benchmark.results.results import Results

LOGGER = logging.getLogger(__name__)


class Figures:
    """The figures dedicated to a group of problem configurations."""

    __algorithm_configurations: AlgorithmsConfigurations
    """The algorithm configurations."""

    __ALPHA: Final[float] = 0.3
    """The opacity level for overlapping areas.
    (Refer to the Matplotlib documentation.)"""

    __directory_path: Path
    """The path to the root directory for the figures."""

    __GRID_SETTINGS: Final[Mapping[str, str]] = {"visible": True, "linestyle": ":"}
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

    __plot_settings: Mapping[str, ConfigurationPlotOptions]
    """The keyword arguments of `matplotlib.axes.Axes.plot`
      for each algorithm configuration."""

    __results: Results
    """The paths to the reference histories
    for each algorithm configuration and problem configuration."""

    __TABLE_PERCENTILES: Final[dict[str, int]] = {
        "maximum": 100,
        "75th centile": 75,
        "median": 50,
        "25th centile": 25,
        "minimum": 0,
    }
    """The percentiles to be displayed in the report tables."""

    __TARGET_VALUES_PLOT_SETTINGS: ClassVar[Mapping[str, str | int | float]] = {
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

    ProblemFigurePaths = dict[_FigureFileName | str, Path | dict[_FigureFileName, Path]]
    """The paths to the figures dedicated to a problem configuration."""

    class _TableFileName(enum.Enum):
        """The name of a table file."""

        INFEASIBILITY_MEASURE = "infeasibility_measure.csv"
        NUMBER_OF_UNSATISFIED_CONSTRAINTS = "number_of_unsatisfied_constraints.csv"
        PERFORMANCE_MEASURE = "performance_measure.csv"

    ProblemTablePaths = dict[_TableFileName | str, Path | dict[_TableFileName, Path]]
    """The paths to the tables dedicated to a problem configuration."""

    def __init__(
        self,
        algorithm_configurations: AlgorithmsConfigurations,
        group: ProblemsGroup,
        results: Results,
        directory_path: Path,
        infeasibility_tolerance: float,
        max_eval_number: int,
        plot_settings: Mapping[str, ConfigurationPlotOptions],
    ) -> None:
        """
        Args:
            algorithm_configurations: The algorithm configurations.
            group: The group of problems.
            results: The paths to the reference histories
                for each algorithm configuration and problem configuration.
            directory_path: The path to the root directory for the figures.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum number of evaluations to be displayed
                on the figures.
                If 0, all the evaluations are displayed.
            plot_settings: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.
        """  # noqa: D205, D212, D415
        self.__algorithm_configurations = algorithm_configurations
        self.__directory_path = directory_path
        self.__group = group
        self.__infeasibility_tolerance = infeasibility_tolerance
        self.__max_eval_number = max_eval_number
        self.__plot_settings = _get_configuration_plot_options(
            plot_settings, algorithm_configurations.names
        )
        self.__results = results

    def plot_data_profiles(self, use_abscissa_log_scale: bool = False) -> Path:
        """Plot the data profiles of the group of problems.

        Args:
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.

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
            plot_settings=self.__plot_settings,
            grid_settings=self.__GRID_SETTINGS,
            use_abscissa_log_scale=use_abscissa_log_scale,
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
        use_abscissa_log_scale: bool,
        table_values_format: str = ".6g",
        bypass_unequal_representation: bool = False,
    ) -> tuple[dict[str, ProblemFigurePaths], dict[str, ProblemTablePaths]]:
        """Plot the figures for each problem configuration of the group.

        Args:
            plot_all_histories: Whether to plot all the performance histories.
            use_performance_log_scale: Whether to use a logarithmic scale
                for the performance measure axis.
            plot_only_median: Whether to plot only the median and no other centile.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
            table_values_format: The string format for the table values.
            bypass_unequal_representation: Whether to bypass the check that ensures
                that each algorithm configuration is represented by the same number of
                performance histories for a given problem configuration.

        Returns:
            The paths to the figures and the paths to the tables.
            The keys are the names of the problems and the values
            are the corresponding dictionaries of figures or tables.

        Raises:
            ValueError: If ``bypass_unequal_representation`` is ``False`` and at least
                one algorithm configuration is represented by fewer performance
                histories than another on an given problem configuration.
        """
        problems_to_figures = {}
        problems_to_tables = {}
        for problem_configuration in self.__group:
            problem_dir = self.__directory_path / join_substrings(
                problem_configuration.name
            )
            problem_dir.mkdir()
            # Gather the performance histories
            performance_histories = {
                algorithm_configuration: PerformanceHistories(*[
                    PerformanceHistory.from_file(path)
                    for path in self.__results.get_paths(
                        algorithm_configuration.name, problem_configuration.name
                    )
                ])
                .cumulate_minimum()
                .get_equal_size_histories()  # FIXME: avoid
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
                    f"'{problem_configuration.name}': {numbers}."
                )
                if bypass_unequal_representation:
                    LOGGER.warning(message)
                else:
                    raise ValueError(message)

            if not problem_configuration.minimize_performance_measure:
                for histories in performance_histories.values():
                    histories.switch_performance_measure_sign()

            # Draw the plots dedicated to each problem configuration.
            problems_to_figures[problem_configuration.name] = (
                self.__get_problem_figures(
                    problem_configuration,
                    performance_histories,
                    problem_dir,
                    plot_all_histories,
                    use_performance_log_scale,
                    plot_only_median,
                    use_time_log_scale,
                    use_abscissa_log_scale,
                )
            )
            # Fill the tables dedicated to each problem configuration.
            problems_to_tables[problem_configuration.name] = self.__get_problem_tables(
                performance_histories,
                problem_dir,
                table_values_format,
                problem_configuration.minimize_performance_measure,
            )

        return problems_to_figures, problems_to_tables

    def __get_problem_figures(
        self,
        problem_configuration: BaseProblemConfiguration,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        plot_all_histories: bool,
        use_performance_log_scale: bool,
        plot_only_median: bool,
        use_time_log_scale: bool,
        use_abscissa_log_scale: bool,
    ) -> ProblemFigurePaths:
        """Return the results figures of a problem configuration.

        Args:
            problem_configuration: The problem configuration.
            performance_histories: The performance histories
                for the problem configuration.
            directory_path: The path to the root directory for the figures.
            plot_all_histories: Whether to plot all the performance histories.
            use_performance_log_scale: Whether to use a logarithmic scale
                for the performance measure axis.
            plot_only_median: Whether to plot only the median and no other centile.
            max_feasible_performance: The maximum feasible performance value.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.

        Returns:
            The paths to the figures dedicated to the problem configuration.
        """
        # Find the extremal feasible performance measure.
        worst_feasible_performances = [
            history.remove_leading_infeasible()[0].performance_measure
            for histories in performance_histories.values()
            for history in histories
            if history[-1].is_feasible
        ]
        infeasible_performance_measure = float("nan")
        if problem_configuration.minimize_performance_measure:
            extremal_feasible_performance = max(
                worst_feasible_performances, default=None
            )
            if extremal_feasible_performance is None:
                infeasible_performance_measure = float("inf")
        else:
            extremal_feasible_performance = min(
                worst_feasible_performances, default=None
            )
            if extremal_feasible_performance is None:
                infeasible_performance_measure = -float("inf")

        maximal_infeasibility = max(
            history[0].infeasibility_measure
            for histories in performance_histories.values()
            for history in histories
        )
        figures = {
            self._FigureFileName.DATA_PROFILE: self.__plot_data_profiles(
                problem_configuration, directory_path, use_abscissa_log_scale
            )
        }
        (
            figures[self._FigureFileName.PERFORMANCE_MEASURE],
            figures[self._FigureFileName.PERFORMANCE_MEASURE_FOCUS],
        ) = self.__plot_performance_measure(
            problem_configuration,
            performance_histories,
            directory_path,
            extremal_feasible_performance,
            infeasible_performance_measure,
            plot_all_histories,
            use_performance_log_scale,
            plot_only_median,
            use_abscissa_log_scale,
            use_time_log_scale,
        )
        figures[self._FigureFileName.EXECUTION_TIME] = self.__plot_execution_time(
            problem_configuration,
            performance_histories,
            directory_path,
            plot_only_median,
            plot_all_histories,
            use_abscissa_log_scale,
            use_time_log_scale,
        )
        if problem_configuration.number_of_scalar_constraints:
            figures[self._FigureFileName.INFEASIBILITY_MEASURE] = (
                self.__plot_infeasibility_measure(
                    problem_configuration,
                    performance_histories,
                    directory_path,
                    maximal_infeasibility,
                    plot_only_median,
                    use_abscissa_log_scale,
                    plot_all_histories,
                    use_time_log_scale,
                )
            )
            figures[self._FigureFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS] = (
                self.__plot_number_of_unsatisfied_constraints(
                    problem_configuration,
                    performance_histories,
                    directory_path,
                    plot_only_median,
                    use_abscissa_log_scale,
                    plot_all_histories,
                    use_time_log_scale,
                )
            )

        figures.update(
            self.__get_algorithms_plots(
                problem_configuration,
                performance_histories,
                extremal_feasible_performance,
                infeasible_performance_measure,
                directory_path,
                use_abscissa_log_scale,
                use_performance_log_scale,
                plot_all_histories,
            )
        )
        return figures

    def __plot_data_profiles(
        self,
        problem_configuration: BaseProblemConfiguration,
        directory_path: Path,
        use_abscissa_log_scale: bool,
    ) -> Path:
        """Plot the data profiles for a problem configuration.

        Args:
            problem_configuration: The problem configuration.
            directory_path: The destination directory for the figure.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.

        Returns:
            The path to the figure.
        """
        if len(self.__group) == 1:
            # Return the path to the data profiles of the group.
            file_path = self.__get_data_profiles_path()
            if file_path.is_file():
                return file_path

            return self.plot_data_profiles(use_abscissa_log_scale)

        file_path = directory_path / self._FigureFileName.DATA_PROFILE.value
        problem_configuration.compute_data_profile(
            self.__algorithm_configurations,
            self.__results,
            False,
            file_path,
            self.__infeasibility_tolerance,
            self.__max_eval_number,
            self.__plot_settings,
            self.__GRID_SETTINGS,
            use_abscissa_log_scale,
        )
        return file_path

    def __plot_performance_measure(
        self,
        problem_configuration: BaseProblemConfiguration,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        extremal_feasible_performance: float,
        infeasible_performance_measure: float,
        plot_all_histories: bool,
        use_performance_log_scale: bool,
        plot_only_median: bool,
        use_abscissa_log_scale: bool,
        use_time_log_scale: bool,
    ) -> tuple[Path, Path]:
        """Plot the performance measure of algorithm configurations on a problem.

        Args:
            problem_configuration: The problem configuration.
            performance_histories: The performance histories
                for the problem configuration.
            directory_path: The path to the root directory for the figures.
            extremal_feasible_performance: The extremal feasible performance measure.
            infeasible_performance_measure: The value to replace the performance measure
                of infeasible history items when computing statistics.
            plot_all_histories: Whether to plot all the performance histories.
            use_performance_log_scale: Whether to use a logarithmic scale
                for the performance measure axis.
            plot_only_median: Whether to plot only the median and no other centile.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.

        Returns:
            The path to the main figure
            and the path to a focus on the target values.
        """
        figsize = matplotlib.rcParams["figure.figsize"]
        figure, axes_array = matplotlib.pyplot.subplots(
            1, 2, sharey="all", figsize=(figsize[0] * 2, figsize[1])
        )
        for axes, abscissa_data_type, use_log_scale in zip(
            axes_array,
            (problem_configuration.abscissa_data_type, TimeAbscissaData),
            (use_abscissa_log_scale, use_time_log_scale),
            strict=False,
        ):
            RangePlot(
                self.__plot_settings,
                self.__GRID_SETTINGS,
                self.__ALPHA,
                self.__MATPLOTLIB_LOG_SCALE,
            ).plot(
                axes,
                performance_histories,
                PerformanceData(
                    axes,
                    infeasible_performance_measure,
                    problem_configuration.performance_measure_label,
                ),
                abscissa_data_type(
                    axes,
                    problem_configuration.number_of_scalar_constraints,
                    use_log_scale,
                ),
                extremal_feasible_performance,
                plot_only_median,
                plot_all_histories,
                problem_configuration.minimize_performance_measure,
                float("inf")
                if problem_configuration.minimize_performance_measure
                else -float("inf"),
                use_performance_log_scale,
            )
            problem_configuration.target_values.plot_on_axes(
                axes,
                self.__TARGET_VALUES_PLOT_SETTINGS,
                set_ylabel_settings={"rotation": 270, "labelpad": 20},
            )

        figure.tight_layout()
        file_path = directory_path / self._FigureFileName.PERFORMANCE_MEASURE.value
        save_show_figure(figure, False, file_path)

        # Plot a focus on the target values
        for performance_axes, targets_axes in (figure.axes[:2], figure.axes[2:]):
            self.__focus_on_targets(
                problem_configuration,
                performance_histories.values(),
                performance_axes,
                targets_axes,
            )

        figure.tight_layout()
        focus_file_path = (
            directory_path / self._FigureFileName.PERFORMANCE_MEASURE_FOCUS.value
        )
        save_show_figure(figure, False, focus_file_path)
        return file_path, focus_file_path

    def __focus_on_targets(
        self,
        problem_configuration: BaseProblemConfiguration,
        performance_histories: Iterable[PerformanceHistories],
        performance_axes: matplotlib.axes.Axes,
        target_axes: matplotlib.axes.Axes,
    ) -> None:
        """Focus a plot on target values.

        Args:
            problem_configuration: The problem configuration.
            performance_histories: The performance histories
                for the problem configuration.
            performance_axes: The axes for the performance measure.
            target_axes: The axes for the target values.
        """
        history_items = [
            history[-1] for histories in performance_histories for history in histories
        ] + list(problem_configuration.minimization_target_values)
        target_items = [
            target
            for target in problem_configuration.minimization_target_values
            if target.is_feasible
        ]
        best_performance_measure = min(history_items).performance_measure
        worst_performance_measure = max(target_items).performance_measure
        if problem_configuration.minimize_performance_measure:
            args = (best_performance_measure, worst_performance_measure)
        else:
            args = (-worst_performance_measure, -best_performance_measure)

        performance_axes.set_ylim(*args)
        target_axes.set_ylim(performance_axes.get_ylim())

    # TODO: remove?
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
            item.performance_measure
            if item.is_feasible
            else infeasible_performance_measure
        )

    def __plot_infeasibility_measure(
        self,
        problem_configuration: BaseProblemConfiguration,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        maximal_infeasibility: float,
        plot_only_median: bool,
        use_abscissa_log_scale: bool,
        plot_all_histories: bool,
        use_time_log_scale: bool,
    ) -> Path:
        """Plot the infeasibility measure of algorithm configurations on a problem.

        Args:
            problem_configuration: The problem configuration.
            performance_histories: The performance histories
                for the problem configuration.
            directory_path: The path to the root directory for the figures.
            maximal_infeasibility: The maximal infeasibility measure.
            plot_only_median: Whether to plot only the median and no other centile.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
            plot_all_histories: Whether to plot all the performance histories.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.

        Returns:
            The path to the figure.
        """
        figsize = matplotlib.rcParams["figure.figsize"]
        figure, axes_array = matplotlib.pyplot.subplots(
            1, 2, sharey="all", figsize=(figsize[0] * 2, figsize[1])
        )
        for axes, abscissa_data_type, use_log_scale in zip(
            axes_array,
            (problem_configuration.abscissa_data_type, TimeAbscissaData),
            (use_abscissa_log_scale, use_time_log_scale),
            strict=False,
        ):
            RangePlot(
                self.__plot_settings,
                self.__GRID_SETTINGS,
                self.__ALPHA,
                self.__MATPLOTLIB_LOG_SCALE,
            ).plot(
                axes,
                performance_histories,
                InfeasibilityData(axes),
                abscissa_data_type(
                    axes,
                    problem_configuration.number_of_scalar_constraints,
                    use_log_scale,
                ),
                maximal_infeasibility,
                plot_only_median,
                plot_all_histories,
                problem_configuration.minimize_performance_measure,
                float("inf"),
                True,
            )

        figure.tight_layout()
        file_path = directory_path / self._FigureFileName.INFEASIBILITY_MEASURE.value
        save_show_figure(figure, False, file_path)
        return file_path

    # TODO: remove?
    @staticmethod
    def __get_infeasibility_measure(item: HistoryItem) -> float:
        """Return the infeasibility measure of a history item."""
        return item.infeasibility_measure

    def __plot_number_of_unsatisfied_constraints(
        self,
        problem_configuration: BaseProblemConfiguration,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        plot_only_median: bool,
        use_abscissa_log_scale: bool,
        plot_all_histories: bool,
        use_time_log_scale: bool,
    ) -> Path:
        """Plot the number of constraints unsatisfied by algorithm configurations.

        Args:
            problem_configuration: The problem configuration.
            performance_histories: The performance histories
                of each algorithm configuration.
            directory_path: The path to the directory where to save the figure.
            plot_only_median: Whether to plot only the median and no other centile.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
            plot_all_histories: Whether to plot all the performance histories.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.

        Returns:
            The path to the figure.
        """
        figsize = matplotlib.rcParams["figure.figsize"]
        figure, axes_array = matplotlib.pyplot.subplots(
            1, 2, sharey="all", figsize=(figsize[0] * 2, figsize[1])
        )
        for axes, abscissa_data_type, use_log_scale in zip(
            axes_array,
            (problem_configuration.abscissa_data_type, TimeAbscissaData),
            (use_abscissa_log_scale, use_time_log_scale),
            strict=False,
        ):
            RangePlot(
                self.__plot_settings,
                self.__GRID_SETTINGS,
                self.__ALPHA,
                self.__MATPLOTLIB_LOG_SCALE,
            ).plot(
                axes,
                performance_histories,
                ConstraintData(axes),
                abscissa_data_type(
                    axes,
                    problem_configuration.number_of_scalar_constraints,
                    use_log_scale,
                ),
                problem_configuration.number_of_scalar_constraints,
                plot_only_median,
                plot_all_histories,
                True,
                problem_configuration.number_of_scalar_constraints,
                False,
            )

        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        figure.tight_layout()
        file_path = (
            directory_path
            / self._FigureFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS.value
        )
        save_show_figure(figure, False, file_path)
        return file_path

    # TODO: remove?
    @staticmethod
    def __get_number_of_unsatisfied_constraints(item: HistoryItem) -> int | float:
        """Return the number of unsatisfied constraints of a history item."""
        number = item.n_unsatisfied_constraints
        return numpy.nan if number is None else number

    def __plot_execution_time(
        self,
        problem_configuration: BaseProblemConfiguration,
        performance_histories: Mapping[AlgorithmConfiguration, PerformanceHistories],
        directory_path: Path,
        plot_only_median: bool,
        plot_all_histories: bool,
        use_abscissa_log_scale: bool,
        use_time_log_scale: bool,
    ) -> Path:
        """Plot the number of constraints unsatisfied by algorithm configurations.

        Args:
            problem_configuration: The problem configuration.
            performance_histories: The performance histories
                of each algorithm configuration.
            directory_path: The path to the directory where to save the figure.
            plot_only_median: Whether to plot only the median and no other centile.
            plot_all_histories: Whether to plot all the performance histories.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.

        Returns:
            The path to the figure.
        """
        figure, axes = matplotlib.pyplot.subplots()
        RangePlot(
            self.__plot_settings,
            self.__GRID_SETTINGS,
            self.__ALPHA,
            self.__MATPLOTLIB_LOG_SCALE,
        ).plot(
            axes,
            performance_histories,
            TimeOrdinateData(axes),
            problem_configuration.abscissa_data_type(
                axes,
                problem_configuration.number_of_scalar_constraints,
                use_abscissa_log_scale,
            ),
            max(
                history.total_time
                for histories in performance_histories.values()
                for history in histories
            ),
            plot_only_median,
            plot_all_histories,
            True,
            float("inf"),
            use_time_log_scale,
        )
        file_path = directory_path / self._FigureFileName.EXECUTION_TIME.value
        save_show_figure(figure, False, file_path)
        return file_path

    def __get_algorithms_plots(
        self,
        problem_configuration: BaseProblemConfiguration,
        performance_histories: Mapping[str, PerformanceHistories],
        extremal_feasible_performance: float | None,
        infeasible_performance_measure: float,
        directory_path: Path,
        use_abscissa_log_scale: bool,
        use_performance_log_scale: bool,
        plot_all_histories: bool,
    ) -> dict[str, dict[_FigureFileName, Path]]:
        """Return the figures associated with algorithm configurations for a problem.

        Args:
            problem_configuration: The problem configuration.
            performance_histories: The performance histories
                for the problem configuration.
            extremal_feasible_performance: The extremal feasible performance measure.
            infeasible_performance_measure: The value to replace the performance measure
                of infeasible history items when computing statistics.
            directory_path: The path to the directory where to save the figures.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
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
                extremal_feasible_performance,
                infeasible_performance_measure,
                plot_all_histories,
                problem_configuration.minimize_performance_measure,
            )
            if use_performance_log_scale:
                axes.set_yscale(self.__MATPLOTLIB_SYMMETRIC_LOG_SCALE)

            if use_abscissa_log_scale:
                axes.set_xscale(self.__MATPLOTLIB_LOG_SCALE)

            axes.grid(**self.__GRID_SETTINGS)
            performance_figures[configuration] = figure

        self.__set_common_limits(performance_figures.values())

        for configuration, figure in performance_figures.items():
            # Add the target values and save the figure
            problem_configuration.target_values.plot_on_axes(
                figure.gca(), self.__TARGET_VALUES_PLOT_SETTINGS
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
            if problem_configuration.minimize_performance_measure:
                performance_axes.set_ylim(
                    top=max(problem_configuration.target_values).performance_measure
                )
            else:
                performance_axes.set_ylim(
                    bottom=min(problem_configuration.target_values).performance_measure
                )

            targets_axes.set_ylim(performance_axes.get_ylim())
            file_path = (
                configuration_dir / self._FigureFileName.PERFORMANCE_MEASURE_FOCUS.value
            )
            save_show_figure(figure, False, file_path)
            figures[configuration.name][
                self._FigureFileName.PERFORMANCE_MEASURE_FOCUS
            ] = file_path

        if problem_configuration.number_of_scalar_constraints:
            # Plot the infeasibility measure distribution for each configuration
            infeasibility_figures = {}
            for configuration in self.__algorithm_configurations:
                figure, axes = matplotlib.pyplot.subplots()
                performance_histories[
                    configuration
                ].plot_infeasibility_measure_distribution(axes, plot_all_histories)
                axes.set_yscale(self.__MATPLOTLIB_LOG_SCALE)
                if use_abscissa_log_scale:
                    axes.set_xscale(self.__MATPLOTLIB_LOG_SCALE)

                axes.grid(**self.__GRID_SETTINGS)
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
                if use_abscissa_log_scale:
                    axes.set_xscale(self.__MATPLOTLIB_LOG_SCALE)

                axes.grid(**self.__GRID_SETTINGS)
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
            The paths to the tables dedicated to the problem configuration.
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
