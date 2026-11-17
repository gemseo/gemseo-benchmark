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
"""Tests for the data profile."""

from __future__ import annotations

import warnings
from datetime import timedelta
from unittest import mock

import pytest

from gemseo_benchmark.data_profiles.data_profile import DataProfile
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.results.history_item import HistoryItem


def test_consistent_target_values():
    """Check the setting of consistent target values."""
    with pytest.raises(
        ValueError,
        match="The reference problems must have the same number of target values",
    ):
        DataProfile({
            "problem_1": TargetValues([1.0, 0.0]),
            "problem_2": TargetValues([2.0]),
        })


def test_add_history_unknown_problem():
    """Check the addition of a history for an unknown problem."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    with pytest.raises(
        ValueError, match="'toto' is not the name of a reference problem"
    ):
        data_profile.add_history("toto", "algo", [2.0, 1.5, 1.0, 0.5, 0.1, 0.0])


def test_compute_data_profiles():
    """Check the computation of data profiles."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo", [2.0, 1.5, 1.0, 0.5, 0.1, 0.0])
    profiles = data_profile.compute_data_profiles()
    assert list(profiles.keys()) == ["algo"]
    assert profiles["algo"] == [0.0, 0.0, 0.5, 0.5, 0.5, 1.0]


@pytest.mark.parametrize(
    "use_abscissa_log_scale", [False, True], ids=["linear_scale", "log_scale"]
)
def test_plot_data_profiles(use_abscissa_log_scale, snapshot_matplotlib):
    """Check the data profiles figure."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo", [2.0, 1.5, 1.0, 0.5, 0.1, 0.0])
    data_profiles = data_profile.compute_data_profiles("algo")
    figure = data_profile._plot_data_profiles(
        data_profiles, use_abscissa_log_scale=use_abscissa_log_scale
    )
    assert figure == snapshot_matplotlib


@pytest.mark.parametrize("converter", [lambda _: _, str])
def test_plot_save(tmp_path, converter):
    """Check the save of the data profiles plot.

    Args:
        converter: The Path converter.
    """  # ruff: ignore[undocumented-param]
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo", [2.0, 1.5, 1.0, 0.5, 0.1, 0.0])
    path = tmp_path / "data_profile.png"
    data_profile.plot(show=False, file_path=converter(path))
    assert path.is_file()


def test_target_values_getter():
    """Check the getting of target values."""
    targets = DataProfile({"problem": TargetValues([1.0, 0.0])}).target_values
    assert len(targets) == 1
    assert len(targets["problem"]) == 2
    assert targets["problem"][0] == HistoryItem(1.0, 0.0)
    assert targets["problem"][1] == HistoryItem(0.0, 0.0)


def test_different_sizes_histories():
    """Check the computation of a data profile based on histories of different sizes."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo", [2.0, 2.0])
    data_profile.add_history("problem", "algo", [2.0, 1.0, 0.0])
    profile = data_profile.compute_data_profiles()
    assert profile["algo"] == [0.0, 0.25, 0.5]


def test_unevenly_represented_problems():
    """Check the handling of unevenly represented reference problems."""
    data_profile = DataProfile({
        "problem1": TargetValues([1.0, 0.0]),
        "problem2": TargetValues([1.0, 0.0]),
    })
    data_profile.add_history("problem1", "algo", [2.0, 2.0])
    data_profile.add_history("problem1", "algo", [2.0, 2.0])
    data_profile.add_history("problem2", "algo", [2.0, 2.0])
    with pytest.raises(
        ValueError,
        match="Reference problems unequally represented for algorithm 'algo'",
    ):
        data_profile.compute_data_profiles()


def test_different_sizes_data_profiles(snapshot_matplotlib):
    """Check the plotting of data profiles of different sizes."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo1", [1.0, 1.0])
    data_profile.add_history("problem", "algo2", [2.0, 1.0, 0.0])
    profiles = data_profile.compute_data_profiles()
    figure = data_profile._plot_data_profiles(profiles)
    assert figure == snapshot_matplotlib(remove_text=True)


def test_plot_algo_names():
    """Check the plotting of the data profiles of some algorithms."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo1", [2.0, 1.0, 0.0])
    data_profile.add_history("problem", "algo2", [2.0, 1.0, 0.0])
    with mock.patch.object(DataProfile, "_plot_data_profiles") as plot_data_profiles:
        data_profile.plot(algo_names=["algo1"], show=False)

    assert plot_data_profiles.call_args.args[0].keys() == {"algo1"}


def _create_data_profile() -> DataProfile:
    """Create a data profile with timed histories of two algorithms.

    Returns:
        A data profile with two timed histories for each of two algorithms.
    """
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history(
        "problem",
        "algo1",
        [2.0, 1.0, 0.0],
        elapsed_times=[timedelta(seconds=t) for t in (1.0, 2.0, 4.0)],
    )
    data_profile.add_history(
        "problem",
        "algo1",
        [2.0, 0.5],
        elapsed_times=[timedelta(seconds=t) for t in (0.5, 3.0)],
    )
    data_profile.add_history(
        "problem",
        "algo2",
        [1.0, 1.0],
        elapsed_times=[timedelta(seconds=t) for t in (0.5, 1.5)],
    )
    data_profile.add_history(
        "problem",
        "algo2",
        [2.0, 0.0],
        elapsed_times=[timedelta(seconds=t) for t in (1.0, 2.5)],
    )
    return data_profile


def test_compute_time_data_profiles():
    """Check the computation of data profiles relative to the elapsed time."""
    profiles = _create_data_profile().compute_time_data_profiles()
    assert profiles == {
        "algo1": ([0.5, 1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.25, 0.5, 0.75]),
        "algo2": ([0.5, 1.0, 1.5, 2.5], [0.25, 0.25, 0.25, 0.75]),
    }


def test_compute_time_data_profiles_algo_names():
    """Check the computation of time data profiles of some algorithms."""
    profiles = _create_data_profile().compute_time_data_profiles("algo2")
    assert profiles.keys() == {"algo2"}


def test_compute_time_data_profiles_end():
    """Check that time and iteration data profiles end with the same ratios."""
    data_profile = _create_data_profile()
    profiles = data_profile.compute_data_profiles()
    time_profiles = data_profile.compute_time_data_profiles()
    for name, (_, ratios) in time_profiles.items():
        assert ratios[-1] == profiles[name][-1]


def test_compute_time_data_profiles_without_elapsed_times():
    """Check the time data profiles of histories without elapsed times."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo", [2.0, 1.5, 1.0, 0.5, 0.1, 0.0])
    assert data_profile.compute_time_data_profiles() == {}


def test_compute_time_data_profiles_with_untimed_history():
    """Check that an algorithm with an untimed history is left out of the profiles."""
    data_profile = _create_data_profile()
    data_profile.add_history("problem", "algo1", [2.0, 0.0])
    assert data_profile.compute_time_data_profiles().keys() == {"algo2"}


def test_plot_time_data_profiles_log_scale_from_zero():
    """Check that a time axis of logarithmic scale is not limited at zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DataProfile._plot_time_data_profiles(
            {"algo": ([0.0, 1.0, 2.0], [0.0, 0.5, 1.0])}, use_time_log_scale=True
        )


@pytest.mark.parametrize(
    "use_time_log_scale", [False, True], ids=["linear_scale", "log_scale"]
)
def test_plot_time_data_profiles(use_time_log_scale, snapshot_matplotlib):
    """Check the figure of the data profiles relative to the elapsed time."""
    data_profile = _create_data_profile()
    profiles = data_profile.compute_time_data_profiles()
    figure = data_profile._plot_time_data_profiles(
        profiles, use_time_log_scale=use_time_log_scale
    )
    assert figure == snapshot_matplotlib


def test_plot_combined_figures(snapshot_matplotlib):
    """Check the figure combining the data profiles."""
    figures = _create_data_profile().plot(show=False, use_subplots=True)
    assert len(figures) == 1
    assert len(figures[0].axes) == 2
    assert figures[0] == snapshot_matplotlib


@pytest.mark.parametrize("show", [False, True], ids=["no_show", "show"])
def test_plot_figures(show):
    """Check the figures of the data profiles."""
    figures = _create_data_profile().plot(show=show)
    assert [figure.axes[0].get_xlabel() for figure in figures] == [
        "Number of evaluations",
        "Elapsed time",
    ]


@pytest.mark.parametrize(
    ("use_subplots", "file_names"),
    [
        (False, {"data_profile_eval.png", "data_profile_time.png"}),
        (True, {"data_profile.png"}),
    ],
    ids=["no_subplots", "subplots"],
)
def test_plot_save_files(tmp_path, use_subplots, file_names):
    """Check the files of the data profiles."""
    _create_data_profile().plot(
        show=False,
        file_path=tmp_path / "data_profile.png",
        use_subplots=use_subplots,
    )
    assert {path.name for path in tmp_path.iterdir()} == file_names


@pytest.mark.parametrize("use_subplots", [False, True], ids=["no_subplots", "subplots"])
def test_plot_without_elapsed_times(tmp_path, use_subplots):
    """Check the data profiles of histories without elapsed times."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo", [2.0, 1.0, 0.0])
    figures = data_profile.plot(
        show=False, file_path=tmp_path / "data_profile.png", use_subplots=use_subplots
    )
    assert len(figures) == 1
    assert len(figures[0].axes) == 1
    assert {path.name for path in tmp_path.iterdir()} == {"data_profile.png"}


def test_plot_options_without_elapsed_times():
    """Check that a configuration keeps its options when another one has no times."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo1", [2.0, 1.0])
    data_profile.add_history(
        "problem",
        "algo2",
        [2.0, 0.0],
        elapsed_times=[timedelta(seconds=t) for t in (1.0, 2.0)],
    )
    figure, time_figure = data_profile.plot(show=False)
    colors = {line.get_label(): line.get_color() for line in figure.axes[0].lines}
    time_colors = {
        line.get_label(): line.get_color() for line in time_figure.axes[0].lines
    }
    assert "algo1" not in time_colors
    assert time_colors["algo2"] == colors["algo2"]
