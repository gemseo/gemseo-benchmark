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
"""Tests for the target values."""
from data_profiles.performance_history import PerformanceHistory
from data_profiles.target_values import TargetValues


def test_count_targets_hist():
    """Check the counting for hit targets"""
    targets = TargetValues([-2.0, 1.0, -1.0], [1.0, 0.0, 0.0])
    history = PerformanceHistory(
        [0.0, -3.0, -1.0, 0.0, 1.0, -1.0], [2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    )
    assert targets.compute_target_hits_history(history) == [0, 0, 0, 2, 2, 3]


def test_plot_save(tmpdir):
    """Check the saving of the target values plot."""
    targets = TargetValues([-2.0, 1.0, -1.0], [1.0, 0.0, 0.0])
    path = tmpdir / "targets.png"
    targets.to_file(str(path))
    assert path.isfile()
