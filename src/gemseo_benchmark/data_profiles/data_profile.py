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
"""Class to compute data profiles for algorithms comparison."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
from gemseo.util.constant import read_only_empty_dict
from gemseo.util.matplotlib_figure import save_show_figure
from numpy import array
from numpy import linspace
from numpy import searchsorted
from numpy import zeros

from gemseo_benchmark import _get_configuration_plot_options
from gemseo_benchmark.results.performance_history import PerformanceHistory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from numbers import Number

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy import ndarray

    from gemseo_benchmark import ConfigurationPlotOptions
    from gemseo_benchmark import MarkeveryType
    from gemseo_benchmark.data_profiles.target_values import TargetValues

TimeDataProfiles = dict[str, tuple[list[float], list[float]]]
"""The data profiles relative to the elapsed time.

For each algorithm configuration,
the elapsed times in seconds and the ratios of targets reached at these times.
"""


class DataProfile:
    """Data profile that compares iterative algorithms on reference problems.

    A data profile is a graphical tool to compare iterative algorithms,
    e.g. optimization algorithms or root-finding algorithms, on reference problems.

    Each of the reference problems must be assigned targets,
    i.e. values of the objective function or values of the residual norm,
    ranging from a first acceptable value to the best known value for the problem.

    The algorithms will be compared based on the number of targets they reach,
    cumulated over all the reference problems,
    relative to the number of problems functions evaluations they make
    or relative to the time they take.

    The data profile is the empirical cumulated distribution function of the number of
    functions evaluations (or of the elapsed time)
    made by an algorithm to reach a problem target.
    """

    __TIME_FILE_NAME_SUFFIX: Final[str] = "_time"
    """The suffix appended to the file name of the data profiles
    relative to the elapsed time."""

    __time_label: ClassVar[str] = "Elapsed time"
    """The label of the time axis."""

    def __init__(self, target_values: Mapping[str, TargetValues]) -> None:
        """
        Args:
            target_values: The target values of each of the reference problems.
        """  # ruff: ignore[missing-blank-line-after-summary, multi-line-summary-first-line, missing-terminal-punctuation]
        self.__targets_number = 0
        self.target_values = target_values
        self.__values_histories = {}

    @property
    def target_values(self) -> dict[str, TargetValues]:
        """The target values of each reference problem.

        Target values are a scale of objective function values,
        ranging from an easily achievable one to the best known value.
        A data profile is computed by counting the number of targets reached by an
        algorithm at each iteration.

        Raises:
            ValueError: If the reference problems have different numbers of target
                values.
        """
        return self.__target_values

    @target_values.setter
    def target_values(self, target_values: Mapping[str, TargetValues]) -> None:
        targets_numbers = {len(pb_targets) for pb_targets in target_values.values()}
        if len(targets_numbers) != 1:
            msg = "The reference problems must have the same number of target values."
            raise ValueError(msg)

        self.__target_values = dict(target_values)
        self.__targets_number = targets_numbers.pop()

    def add_history(
        self,
        problem_name: str,
        algorithm_configuration_name: str,
        performance_measures: Sequence[float],
        infeasibility_measures: Sequence[float] | None = None,
        feasibility_statuses: Sequence[bool] | None = None,
        elapsed_times: Sequence[datetime.timedelta] = (),
    ) -> None:
        """Add a history of performance values.

        Args:
            problem_name: The name of the problem.
            algorithm_configuration_name: The name of the algorithm configuration.
            performance_measures: A history of performance measures.
                N.B. the value at index `i` is assumed to have been obtained with
                `i+1` evaluations.
            infeasibility_measures: A history of infeasibility measures.
                If `None` then measures are set to zero in case of feasibility and set
                to infinity otherwise.
            feasibility_statuses: A history of (boolean) feasibility statuses.
                If `None` then feasibility is always assumed.
            elapsed_times: A history of elapsed times, in increasing order.
                If empty, the elapsed times are set to zero.

        Raises:
            ValueError: If the problem name is not the name of a reference problem.
        """
        if problem_name not in self.__target_values:
            msg = f"{problem_name!r} is not the name of a reference problem"
            raise ValueError(msg)
        if algorithm_configuration_name not in self.__values_histories:
            self.__values_histories[algorithm_configuration_name] = {
                pb_name: [] for pb_name in self.__target_values
            }
        history = PerformanceHistory(
            performance_measures,
            infeasibility_measures,
            feasibility_statuses,
            elapsed_times=elapsed_times,
        )
        self.__values_histories[algorithm_configuration_name][problem_name].append(
            history
        )

    # TODO: remove argument 'markevery' in favor of 'plot_settings' (API break)
    def plot(
        self,
        algo_names: Iterable[str] | None = None,
        show: bool = True,
        file_path: str | Path = "",
        markevery: MarkeveryType | None = None,
        plot_settings: Mapping[str, ConfigurationPlotOptions] = read_only_empty_dict,
        grid_settings: Mapping[str, str] = read_only_empty_dict,
        use_abscissa_log_scale: bool = False,
        use_time_log_scale: bool = False,
        combine_figures: bool = False,
    ) -> list[Figure]:
        """Plot the data profiles of the required algorithms.

        The data profiles are plotted
        relative to the number of functions evaluations
        and relative to the elapsed time.

        Args:
            algo_names: The names of the algorithms.
                If `None` then all the algorithms are considered.
            show: If True, show the plot.
            file_path: The path where to save the plot.
                If empty, the plot is not saved.
                Unless `combine_figures` is `True`,
                this path is the one of the data profiles
                relative to the number of functions evaluations
                and the data profiles relative to the elapsed time are saved
                next to it, with the suffix `_time` appended to the file name,
                e.g. `data_profile_time.png` for `data_profile.png`.
            markevery: The sampling parameter for the markers of the plot.
                Refer to the Matplotlib documentation.
            plot_settings: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.
            grid_settings: The keyword arguments of `matplotlib.pyplot.grid`.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.
            combine_figures: Whether to plot the data profiles
                relative to the number of functions evaluations
                and relative to the elapsed time
                side by side in a single figure.

        Returns:
            The figure of the data profiles
            relative to the number of functions evaluations
            and the figure of the data profiles relative to the elapsed time,
            or the single figure gathering both if `combine_figures` is `True`.
        """
        if algo_names is None:
            algo_names = ()

        data_profiles = self.compute_data_profiles(*algo_names)
        time_data_profiles = self.compute_time_data_profiles(*algo_names)
        plot_settings_copy = plot_settings.copy()
        for settings in plot_settings_copy.values():
            if "markevery" not in settings:
                settings["markevery"] = markevery

        if combine_figures:
            figsize = matplotlib.rcParams["figure.figsize"]
            figure, (axes, time_axes) = plt.subplots(
                1, 2, sharey="all", figsize=(figsize[0] * 2, figsize[1])
            )
            self._plot_data_profiles(
                data_profiles,
                plot_settings_copy,
                grid_settings,
                use_abscissa_log_scale,
                axes,
            )
            self._plot_time_data_profiles(
                time_data_profiles,
                plot_settings_copy,
                grid_settings,
                use_time_log_scale,
                time_axes,
            )
            time_axes.set_ylabel("")
            figure.tight_layout()
            figures_and_paths = [(figure, file_path)]
        else:
            time_file_path = ""
            if file_path:
                file_path = Path(file_path)
                time_file_path = file_path.with_stem(
                    f"{file_path.stem}{self.__TIME_FILE_NAME_SUFFIX}"
                )

            figures_and_paths = [
                (
                    self._plot_data_profiles(
                        data_profiles,
                        plot_settings_copy,
                        grid_settings,
                        use_abscissa_log_scale,
                    ),
                    file_path,
                ),
                (
                    self._plot_time_data_profiles(
                        time_data_profiles,
                        plot_settings_copy,
                        grid_settings,
                        use_time_log_scale,
                    ),
                    time_file_path,
                ),
            ]

        for figure, path in figures_and_paths:
            save_show_figure(figure, False, path, close=False)

        if show:
            plt.show()

        figures = [figure for figure, _ in figures_and_paths]
        if file_path:
            for figure in figures:
                plt.close(figure)

        return figures

    def compute_data_profiles(self, *algo_names: str) -> dict[str, list[Number]]:
        """Compute the data profiles of the required algorithms.

        For each algorithm, compute the cumulative distribution function of the number
        of evaluations required by the algorithm to reach a reference target.

        Args:
            algo_names: The names of the algorithms.
                If `None` then all the algorithms are considered.

        Returns:
            The data profiles.
        """
        data_profiles = {}
        if not algo_names:
            algo_names = self.__values_histories.keys()

        for name in algo_names:
            total_hits_history = self.__compute_hits_history(name)
            ratios = total_hits_history / self.__get_targets_total(name)
            data_profiles[name] = ratios.tolist()
        return data_profiles

    def compute_time_data_profiles(self, *algo_names: str) -> TimeDataProfiles:
        """Compute the data profiles of the required algorithms relative to time.

        For each algorithm, compute the cumulative distribution function of the
        elapsed time required by the algorithm to reach a reference target.

        Args:
            *algo_names: The names of the algorithms.
                If empty, all the algorithms are considered.

        Returns:
            The data profiles relative to the elapsed time.
        """
        if not algo_names:
            algo_names = self.__values_histories.keys()

        data_profiles = {}
        for name in algo_names:
            algo_histories = self.__values_histories[name]
            times = sorted({
                item.elapsed_time.total_seconds()
                for pb_histories in algo_histories.values()
                for pb_history in pb_histories
                for item in pb_history
            })
            total_hits_history = zeros(len(times))
            for pb_name, targets in self.__target_values.items():
                for pb_history in algo_histories[pb_name]:
                    # Prepend 0 for the times preceding the first history item.
                    hits_history = array([
                        0,
                        *targets.compute_target_hits_history(pb_history),
                    ])
                    # The number of history items obtained at each time.
                    numbers_of_items = searchsorted(
                        [item.elapsed_time.total_seconds() for item in pb_history],
                        times,
                        side="right",
                    )
                    total_hits_history += hits_history[numbers_of_items]

            ratios = total_hits_history / self.__get_targets_total(name)
            data_profiles[name] = (times, ratios.tolist())

        return data_profiles

    def __get_targets_total(self, algo_name: str) -> int:
        """Return the total number of targets to be reached by an algorithm.

        Args:
            algo_name: The name of the algorithm.

        Returns:
            The total number of targets.
        """
        return (
            self.__targets_number
            * len(self.__target_values)
            * self.__get_repeat_number(algo_name)
        )

    def __compute_hits_history(self, algo_name: str) -> ndarray:
        """Compute the history of the number of target hits of an algorithm.

        Args:
            algo_name: The name of the algorithm.

        Returns:
            The history of the number of target hits.
        """
        algo_histories = self.__values_histories[algo_name]

        # Compute the maximal size of an optimization history
        max_history_size = max(
            max(len(pb_history) for pb_history in algo_history)
            for algo_history in algo_histories.values()
        )

        # Compute the history of the number of target hits across all optimizations
        total_hits_history = zeros(max_history_size)
        for pb_name, targets in self.__target_values.items():
            for pb_history in algo_histories[pb_name]:
                hits_history = targets.compute_target_hits_history(pb_history)
                # If the history is shorter than the longest one, repeat its last value
                if len(hits_history) < max_history_size:
                    tail = [hits_history[-1]] * (max_history_size - len(hits_history))
                    hits_history.extend(tail)

                total_hits_history += array(hits_history)

        return total_hits_history

    def __get_repeat_number(self, algo_name: str) -> int:
        """Check that an algorithm has the same number of histories for each problem.

        Make sure that the reference problems are equally represented with respect to
        the algorithm performance.

        Args:
            algo_name: The name of the algorithm.

        Returns:
            The common number of values histories per problem.

        Raises:
            ValueError: If the algorithm does not have the same number of histories
                for each problem.
        """
        histories_numbers = {
            len(histories) for histories in self.__values_histories[algo_name].values()
        }
        if len(histories_numbers) != 1:
            msg = (
                f"Reference problems unequally represented for algorithm {algo_name!r}."
            )
            raise ValueError(msg)
        return histories_numbers.pop()

    @staticmethod
    def _plot_data_profiles(
        data_profiles: Mapping[str, Sequence[Number]],
        plot_settings: Mapping[str, ConfigurationPlotOptions] = read_only_empty_dict,
        grid_settings: Mapping[str, str] = read_only_empty_dict,
        use_abscissa_log_scale: bool = False,
        axes: Axes | None = None,
    ) -> Figure:
        """Plot the data profiles.

        Args:
            data_profiles: The data profiles.
            plot_settings: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.
            grid_settings: The keyword arguments of `matplotlib.pyplot.grid`.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
            axes: The axes of the plot.
                If `None`, the data profiles are plotted on a new figure.

        Returns:
            The data profiles figure.
        """
        if axes is None:
            axes = plt.figure().add_subplot(1, 1, 1)

        DataProfile.__set_title(axes, data_profiles)
        max_profile_size = max(len(profile) for profile in data_profiles.values())
        axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if use_abscissa_log_scale:
            axes.set_xscale("log")

        axes.set_xlabel("Number of functions evaluations")
        axes.set_xlim([1, max_profile_size])
        DataProfile.__set_ratio_axis(axes)
        DataProfile.__draw_profiles(
            axes,
            {
                name: (range(1, len(profile) + 1), profile)
                for name, profile in data_profiles.items()
            },
            max_profile_size,
            _get_configuration_plot_options(plot_settings, data_profiles),
            grid_settings,
        )
        return axes.figure

    @staticmethod
    def _plot_time_data_profiles(
        data_profiles: TimeDataProfiles,
        plot_settings: Mapping[str, ConfigurationPlotOptions] = read_only_empty_dict,
        grid_settings: Mapping[str, str] = read_only_empty_dict,
        use_time_log_scale: bool = False,
        axes: Axes | None = None,
    ) -> Figure:
        """Plot the data profiles relative to the elapsed time.

        Args:
            data_profiles: The data profiles relative to the elapsed time.
            plot_settings: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.
            grid_settings: The keyword arguments of `matplotlib.pyplot.grid`.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.
            axes: The axes of the plot.
                If `None`, the data profiles are plotted on a new figure.

        Returns:
            The data profiles figure.
        """
        create_figure = axes is None
        if create_figure:
            axes = plt.figure().add_subplot(1, 1, 1)

        DataProfile.__set_title(axes, data_profiles)
        if use_time_log_scale:
            axes.set_xscale("log")
        else:
            axes.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(
                    lambda seconds, _: str(datetime.timedelta(seconds=seconds))
                )
            )

        axes.tick_params(axis="x", labelrotation=90)
        axes.set_xlabel(DataProfile.__time_label)
        min_time = min(times[0] for times, _ in data_profiles.values())
        max_time = max(times[-1] for times, _ in data_profiles.values())
        if min_time < max_time:
            axes.set_xlim([min_time, max_time])

        DataProfile.__set_ratio_axis(axes)
        DataProfile.__draw_profiles(
            axes,
            data_profiles,
            max_time,
            {
                name: {"drawstyle": "steps-post", **settings}
                for name, settings in _get_configuration_plot_options(
                    plot_settings, data_profiles
                ).items()
            },
            grid_settings,
        )
        if create_figure:
            # Make room for the rotated time tick labels.
            axes.figure.tight_layout()

        return axes.figure

    @staticmethod
    def __set_title(axes: Axes, data_profiles: Mapping[str, object]) -> None:
        """Set the title of a data profiles plot.

        Args:
            axes: The axes of the plot.
            data_profiles: The data profiles.
        """
        axes.set_title(f"Data profile{'s' if len(data_profiles) > 1 else ''}")

    @staticmethod
    def __set_ratio_axis(axes: Axes) -> None:
        """Set the axis of the ratios of targets reached.

        Args:
            axes: The axes of the plot.
        """
        y_ticks = linspace(0.0, 1.0, 11)
        axes.set_yticks(y_ticks, [f"{ratio * 100.0:.0f}%" for ratio in y_ticks])
        axes.set_ylabel("Ratios of targets reached")
        axes.set_ylim([0.0, 1.05])

    @staticmethod
    def __draw_profiles(
        axes: Axes,
        data_profiles: Mapping[str, tuple[Sequence[float], Sequence[float]]],
        max_abscissa: float,
        plot_settings: Mapping[str, ConfigurationPlotOptions],
        grid_settings: Mapping[str, str],
    ) -> None:
        """Draw data profiles.

        A data profile ending before the maximum abscissa
        is extended with a horizontal dotted line
        and its last entry is marked with a star.

        Args:
            axes: The axes of the plot.
            data_profiles: The abscissas and the ratios of each data profile.
            max_abscissa: The maximum abscissa of the data profiles.
            plot_settings: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.
            grid_settings: The keyword arguments of `matplotlib.pyplot.grid`.
        """
        for name, (abscissas, profile) in data_profiles.items():
            axes.plot(abscissas, profile, **plot_settings[name])
            last_abscissa = abscissas[-1]
            if last_abscissa < max_abscissa:
                color = plot_settings[name]["color"]
                last_value = profile[-1]
                axes.plot(
                    [last_abscissa, max_abscissa],
                    [last_value, last_value],
                    color=color,
                    linestyle="dotted",
                )
                axes.plot(last_abscissa, last_value, marker="*", color=color)

        axes.legend()
        axes.grid(**grid_settings)
