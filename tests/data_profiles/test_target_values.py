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

import numpy
import pytest
from matplotlib import pyplot
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


def _count_target_hits(targets: TargetValues, history: PerformanceHistory) -> list[int]:
    """Count the targets hit by a history by comparing every item with every target.

    Args:
        targets: The targets.
        history: The history.

    Returns:
        The history of the number of target hits.
    """
    return [
        [minimum <= target for target in targets].count(True)
        for minimum in history.compute_cumulated_minimum()
    ]


@pytest.mark.parametrize("seed", range(10))
def test_count_targets_hist_random(seed):
    """Check the counting of target hits against a pairwise comparison."""
    rng = numpy.random.default_rng(seed)
    # Few distinct values to get ties between the items and the targets.
    targets = TargetValues(
        rng.integers(-3, 4, 5).tolist(), rng.choice([0.0, 0.0, 1.0, 2.0], 5).tolist()
    )
    history = PerformanceHistory(
        rng.integers(-3, 4, 30).tolist(), rng.choice([0.0, 1.0, 2.0, 3.0], 30).tolist()
    )
    assert targets.compute_target_hits_history(history) == _count_target_hits(
        targets, history
    )


def test_count_targets_hist_nan():
    """Check the counting of target hits with a NaN performance measure."""
    targets = TargetValues([2.0, 1.0, 0.0])
    history = PerformanceHistory([3.0, float("nan"), 1.0, 0.0])
    assert targets.compute_target_hits_history(history) == _count_target_hits(
        targets, history
    )


@pytest.fixture(scope="module")
def feasible_target_values() -> TargetValues:
    """Return feasible target value."""
    return TargetValues([2.0, 1.0, 0.0])


def test_plot_targets(feasible_target_values, snapshot_matplotlib):
    """Check the target values figure."""
    figure = feasible_target_values.plot(show=False)
    assert figure == snapshot_matplotlib(remove_text=True)


@pytest.mark.parametrize("converter", [lambda _: _, str])
def test_plot_save(tmp_path, converter):
    """Check the saving of the target values plot.

    Args:
        converter: The Path converter.
    """  # ruff: ignore[undocumented-param]
    targets = TargetValues([-2.0, 1.0, -1.0], [1.0, 0.0, 0.0])
    path = tmp_path / "targets.png"
    targets.plot(show=False, file_path=converter(path))
    assert path.is_file()


def test_plot_on_axes(feasible_target_values, snapshot_matplotlib) -> None:
    """Check the plotting of target value on specific axes."""
    figure = pyplot.figure()
    axes = figure.gca()
    axes.plot(range(1, 6), range(4, -1, -1))
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    feasible_target_values.plot_on_axes(axes)
    assert figure == snapshot_matplotlib


def test_plot_infeasible_targets():
    """Check the plotting of target values that are all infeasible."""
    figure = TargetValues([2.0, 1.0], [1.0, 1.0]).plot(show=False)
    assert [line.get_label() for line in figure.gca().lines] == ["infeasible"]
