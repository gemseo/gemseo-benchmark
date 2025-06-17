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
"""Tests for the targets generator."""

from __future__ import annotations

import re
from unittest import mock

import pytest
from matplotlib import pyplot
from matplotlib.testing.decorators import image_comparison

from gemseo_benchmark.data_profiles.targets_generator import TargetsGenerator
from gemseo_benchmark.results.history_item import HistoryItem


def test_add_inconsistent_histories():
    """Check the addition of inconsistent performance histories."""
    generator = TargetsGenerator()
    with pytest.raises(ValueError):
        generator.add_history([3.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        generator.add_history([3.0, 2.0], feasibility_statuses=[False])


@pytest.mark.parametrize(
    ("performance_measures", "history", "message"),
    [
        (
            (3.0, 2.0),
            mock.Mock(),
            "Both a performance history and objective values were passed.",
        ),
        (
            None,
            None,
            "Either a performance history or objective values must be passed.",
        ),
    ],
)
def test_add_history_redundant_arguments(performance_measures, history, message):
    """Check the addition of a performance history with redundant arguments."""
    generator = TargetsGenerator()
    with pytest.raises(ValueError, match=message):
        generator.add_history(performance_measures, history=history)


def test_negative_infeasibility_measures():
    """Check the addition of a history with negative infeasibility measures."""
    generator = TargetsGenerator()
    with pytest.raises(ValueError):
        generator.add_history([3.0, 2.0], [1.0, -1.0])


def test_too_many_targets():
    """Check that requiring more targets than are iterations raises an exception."""
    generator = TargetsGenerator()
    generator.add_history([3.0, 2.0])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of targets required (3) is greater than the size the "
            "longest history (2) starting from budget_min (1)."
        ),
    ):
        generator.compute_target_values(3)


def test_infeasible_targets():
    """Check the computation of infeasible targets."""
    generator = TargetsGenerator()
    generator.add_history([3.0, 2.0], [0.0, 1.0])
    generator.add_history([2.0, 1.0], [1.0, 1.0])
    targets = generator.compute_target_values(1, feasible=False, show=False)
    assert targets.items == [HistoryItem(3.0, 0.0)]


def test_various_lengths_histories():
    """Check the computation of targets out of histories of various sizes."""
    generator = TargetsGenerator()
    generator.add_history([3.0, 2.0])
    generator.add_history([2.0])
    targets = generator.compute_target_values(1, show=False)
    assert targets.items == [HistoryItem(2.0, 0.0)]


def test_run(tmp_path):
    """Check the computation of target values."""
    generator = TargetsGenerator()
    generator.add_history([3.0, 2.0])
    generator.add_history([2.0, 3.0])
    generator.add_history([1.0, 0.0])
    generator.add_history([0.0, 2.0])
    generator.add_history([3.0, 0.0])
    path = tmp_path / "targets.png"
    targets = generator.compute_target_values(2, show=False, file_path=path)
    assert targets.items == [HistoryItem(1.0, 0.0), HistoryItem(0.0, 0.0)]
    assert path.is_file()


def test_no_histories():
    """Check the computation of target values without histories."""
    generator = TargetsGenerator()
    with pytest.raises(
        RuntimeError,
        match=re.escape("There are no histories to generate the targets from."),
    ):
        generator.compute_target_values(2, show=False)


@pytest.mark.parametrize("best_target_objective", [None, 0.0])
def test_best_target(best_target_objective):
    """Check the setting of the best target value."""
    generator = TargetsGenerator()
    generator.add_history([2.0, 1.0])
    generator.add_history([1.0, 0.0])
    targets = generator.compute_target_values(
        2, show=False, best_target_objective=best_target_objective
    )
    # Check that only the second history (reaching the best target) is kept
    assert targets.items == [HistoryItem(1.0, 0.0), HistoryItem(0.0, 0.0)]


def test_infeasible_best_target():
    """Check the handling of an infeasible best target."""
    generator = TargetsGenerator()
    generator.add_history([2.0, 1.0], [1.0, 1.0])
    generator.add_history([1.0, 0.0], [1.0, 1.0])
    with pytest.raises(
        RuntimeError, match=re.escape("The best target value is not feasible.")
    ):
        generator.compute_target_values(2, show=False)


def test_best_target_not_reached():
    """Check the case when no history reaches the best target value."""
    generator = TargetsGenerator()
    generator.add_history([2.0, 1.0])
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "There is no performance history that reaches the best target value."
        ),
    ):
        generator.compute_target_values(2, show=False, best_target_objective=0.0)


@pytest.mark.parametrize(
    ("feasibility_statuses", "baseline_images"),
    [
        ([True, True], ["plot_histories"]),
        ([False, True], ["plot_partially_infeasible_history"]),
        ([False, False], ["plot_infeasible_history"]),
    ],
)
@image_comparison(baseline_images=None, remove_text=True, extensions=["png"])
def test_plot_histories(feasibility_statuses, baseline_images):
    """Check the plotting of histories."""
    generator = TargetsGenerator()
    generator.add_history([2.0, 1.0])
    generator.add_history([3.0, 0.0], feasibility_statuses=feasibility_statuses)
    pyplot.close("all")
    generator.plot_histories(best_target_value=0.0, show=False)
