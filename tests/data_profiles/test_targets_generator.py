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
"""Tests for the targets generator."""
from gemseo_benchmark.data_profiles.history_item import HistoryItem
from gemseo_benchmark.data_profiles.targets_generator import TargetsGenerator
from pytest import raises


def test_add_inconsistent_histories():
    """Check the addition of inconsistent performance histories."""
    generator = TargetsGenerator()
    with raises(ValueError):
        generator.add_history([3.0, 2.0], [1.0])
    with raises(ValueError):
        generator.add_history([3.0, 2.0], feasibility_statuses=[False])


def test_negative_infeasibility_measures():
    """Check the addition of a history with negative infeasibility measures."""
    generator = TargetsGenerator()
    with raises(ValueError):
        generator.add_history([3.0, 2.0], [1.0, -1.0])


def test_too_many_targets():
    """Check that requiring more targets than are iterations raises an exception."""
    generator = TargetsGenerator()
    generator.add_history([3.0, 2.0])
    with raises(
            ValueError,
            match="The number of budgets required cannot be larger than "
                  "the number of budgets available: 3 > 2"
    ):
        generator.run(3)


def test_infeasible_targets():
    """Check the computation of infeasible targets."""
    generator = TargetsGenerator()
    generator.add_history([3.0, 2.0], [0.0, 1.0])
    generator.add_history([2.0, 1.0], [1.0, 1.0])
    targets = generator.run(2, feasible=False, show=False)
    assert targets.history_items == [HistoryItem(3.0, 0.0), HistoryItem(3.0, 0.0)]


def test_various_lengths_histories():
    """Check the computation of targets out of histories of various sizes."""
    generator = TargetsGenerator()
    generator.add_history([3.0, 2.0])
    generator.add_history([2.0])
    targets = generator.run(2, show=False)
    assert targets.history_items == [HistoryItem(2.0, 0.0), HistoryItem(2.0, 0.0)]


def test_run(tmpdir):
    """Check the computation of target values."""
    generator = TargetsGenerator()
    generator.add_history([3.0, 2.0])
    generator.add_history([2.0, 3.0])
    generator.add_history([1.0, 0.0])
    path = tmpdir / "targets.png"
    targets = generator.run(2, show=False, path=path)
    assert targets.history_items == [HistoryItem(2.0, 0.0), HistoryItem(2.0, 0.0)]
    assert path.isfile()
