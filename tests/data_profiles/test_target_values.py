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
"""Tests for the target values."""

from __future__ import annotations

import pytest
from matplotlib import pyplot
from matplotlib.testing.decorators import image_comparison
from matplotlib.ticker import MaxNLocator

from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.results.performance_history import PerformanceHistory


def test_count_targets_hist():
    """Check the counting for hit targets."""
    targets = TargetValues([-2.0, 1.0, -1.0], [1.0, 0.0, 0.0])
    history = PerformanceHistory(
        [0.0, -3.0, -1.0, 0.0, 1.0, -1.0], [2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    )
    assert targets.compute_target_hits_history(history) == [0, 0, 0, 2, 2, 3]


@pytest.fixture(scope="module")
def feasible_target_values() -> TargetValues:
    """Return feasible target value."""
    return TargetValues([2.0, 1.0, 0.0])


@image_comparison(baseline_images=["targets"], remove_text=True, extensions=["png"])
def test_plot_targets(feasible_target_values):
    """Check the target values figure."""
    pyplot.close("all")
    feasible_target_values.plot(show=False)


@pytest.mark.parametrize("converter", [lambda _: _, str])
def test_plot_save(tmp_path, converter):
    """Check the saving of the target values plot.

    Args:
        converter: The Path converter.
    """  # noqa: D417
    targets = TargetValues([-2.0, 1.0, -1.0], [1.0, 0.0, 0.0])
    path = tmp_path / "targets.png"
    targets.plot(show=False, file_path=converter(path))
    assert path.is_file()


@image_comparison(baseline_images=["feasible_targets"], extensions=["png"])
def test_plot_on_axes(feasible_target_values) -> None:
    """Check the plotting of target value on specific axes."""
    axes = pyplot.figure().gca()
    axes.plot(range(1, 6), range(4, -1, -1))
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    feasible_target_values.plot_on_axes(axes)
