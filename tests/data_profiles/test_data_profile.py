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
"""Tests for the data profile."""
import pytest
from matplotlib.testing.decorators import image_comparison
from pytest import raises

from data_profiles.data_profile import DataProfile
from data_profiles.target_values import TargetValues


def test_target_values_as_mapping():
    """Check the setting of target values as a mapping."""
    with raises(TypeError, match="The target values be must passed as a mapping"):
        DataProfile([TargetValues([2.0, 1.0])])


def test_consistent_target_values():
    """Check the setting of consistent target values."""
    with raises(ValueError, match="The reference problems must have the same number "
                                  "of target values"):
        DataProfile({
            "problem_1": TargetValues([1.0, 0.0]),
            "problem_2": TargetValues([2.0]),
        })


def test_add_history_unknown_problem():
    """Check the addition of a history for an unknown problem."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    with raises(ValueError, match="toto is not the name of a reference problem"):
        data_profile.add_history("toto", "algo", [2.0, 1.5, 1.0, 0.5, 0.1, 0.0])


def test_compute_data_profiles():
    """Check the computation of data profiles."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo", [2.0, 1.5, 1.0, 0.5, 0.1, 0.0])
    profiles = data_profile.compute_data_profiles()
    assert list(profiles.keys()) == ["algo"]
    assert profiles["algo"] == [0.0, 0.0, 0.5, 0.5, 0.5, 1.0]


@image_comparison(
    baseline_images=["data_profile"], remove_text=True, extensions=['png']
)
def test_plot_data_profiles():
    """Check the data profiles figure."""
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo", [2.0, 1.5, 1.0, 0.5, 0.1, 0.0])
    data_profiles = data_profile.compute_data_profiles("algo")
    data_profile._plot_data_profiles(data_profiles)


@pytest.mark.parametrize("converter", [lambda _: _, str])
def test_plot_save(tmpdir, converter):
    """Check the save of the data profiles plot.

    Args:
        converter: The Path converter.
    """
    data_profile = DataProfile({"problem": TargetValues([1.0, 0.0])})
    data_profile.add_history("problem", "algo", [2.0, 1.5, 1.0, 0.5, 0.1, 0.0])
    path = tmpdir / "data_profile.png"
    data_profile.plot(show=False, path=converter(path))
    assert path.isfile()
