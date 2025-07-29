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

import pytest
from matplotlib import pyplot
from matplotlib.testing.decorators import image_comparison

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
    ("baseline_images", "use_abscissa_log_scale"),
    [
        (
            [f"data_profile[use_abscissa_log_scale={use_abscissa_log_scale}]"],
            use_abscissa_log_scale,
        )
        for use_abscissa_log_scale in [False, True]
    ],
)
@image_comparison(None, ["png"])
def test_plot_data_profiles(baseline_images, use_abscissa_log_scale):
    """Check the data profiles figure."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo", [2.0, 1.5, 1.0, 0.5, 0.1, 0.0])
    data_profiles = data_profile.compute_data_profiles("algo")
    pyplot.close("all")
    data_profile._plot_data_profiles(
        data_profiles, use_abscissa_log_scale=use_abscissa_log_scale
    )


@pytest.mark.parametrize("converter", [lambda _: _, str])
def test_plot_save(tmp_path, converter):
    """Check the save of the data profiles plot.

    Args:
        converter: The Path converter.
    """  # noqa: D417
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


@image_comparison(
    baseline_images=["two_data_profiles"], remove_text=True, extensions=["png"]
)
def test_different_sizes_data_profiles():
    """Check the plotting of data profiles of different sizes."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo1", [1.0, 1.0])
    data_profile.add_history("problem", "algo2", [2.0, 1.0, 0.0])
    profiles = data_profile.compute_data_profiles()
    pyplot.close("all")
    data_profile._plot_data_profiles(profiles)
