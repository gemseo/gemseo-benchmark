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
"""Tests for the performance history."""
from pathlib import Path

from numpy import inf
from pytest import raises

from data_profiles.history_item import HistoryItem
from data_profiles.performance_history import PerformanceHistory


def test_invalid_init_lengths():
    """Check the initialization of a history with lists of inconsistent lengths."""
    with raises(ValueError):
        PerformanceHistory([3.0, 2.0], [1.0])
    with raises(ValueError):
        PerformanceHistory([3.0, 2.0], feasibility_statuses=[False])


def test_negative_infeasibility_measures():
    """Check the initialization of a history with negative infeasibility measures."""
    with raises(ValueError):
        PerformanceHistory([3.0, 2.0], [1.0, -1.0])


def test_length():
    """Check the length of a performance history"""
    history_1 = PerformanceHistory([3.0, 2.0])
    assert len(history_1) == 2
    history_2 = PerformanceHistory([3.0, 2.0], [1.0, 0.0])
    assert len(history_2) == 2
    history_3 = PerformanceHistory([3.0, 2.0], feasibility_statuses=[False, True])
    assert len(history_3) == 2


def test_iter():
    """Check the iteration over a performance history"""
    history = PerformanceHistory([3.0, 2.0], [1.0, 0.0])
    assert list(iter(history)) == [HistoryItem(3.0, 1.0), HistoryItem(2.0, 0.0)]
    history = PerformanceHistory([3.0, 2.0], feasibility_statuses=[False, True])
    assert list(iter(history)) == [HistoryItem(3.0, inf), HistoryItem(2.0, 0.0)]


def test_compute_cumulated_minimum():
    """Check the computation of the cumulated minimum of a performance history"""
    history = PerformanceHistory(
        [0.0, -3.0, -1.0, 0.0, 1.0, -1.0],
        [2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    )
    reference = PerformanceHistory(
        [0.0, 0.0, -1.0, 0.0, 0.0, -1.0], [2.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    )
    cumulated_minimum = history.compute_cumulated_minimum()
    assert cumulated_minimum.history_items == reference.history_items


def test_compute_median_history():
    """Check the computation of the median history."""
    hist_1 = PerformanceHistory([1.0, -1.0, 0.0], [2.0, 0.0, 3.0])
    hist_2 = PerformanceHistory([-2.0, -2.0, 2.0], [0.0, 3.0, 0.0])
    hist_3 = PerformanceHistory([3.0, -3.0, 3.0], [0.0, 0.0, 0.0])
    reference = PerformanceHistory([3.0, -1.0, 3.0], [0.0, 0.0, 0.0])
    median = PerformanceHistory.compute_median_history([hist_1, hist_2, hist_3])
    assert median.history_items == reference.history_items


def test_remove_leading_infeasible():
    """Check the removal of the leading infeasible items in a performance history."""
    history = PerformanceHistory([2.0, 1.0, 0.0, 1.0, -1.0], [2.0, 1.0, 0.0, 3.0, 0.0])
    reference = PerformanceHistory([0.0, 1.0, -1.0], [0.0, 3.0, 0.0])
    truncation = history.remove_leading_infeasible()
    assert truncation.history_items == reference.history_items


def test_save_to_file(tmpdir):
    """Check the writing of a performance history into a file."""
    history = PerformanceHistory([-2.0, -3.0], [1.0, 0.0])
    file_path = tmpdir / "history.json"
    history.save_to_file(str(file_path))
    with file_path.open("r") as file:
        contents = file.read()
    reference_path = Path(__file__).parent / "reference_history.json"
    with reference_path.open("r") as reference_file:
        reference = reference_file.read()
    assert contents == reference


def test_from_file():
    """Check the initialization of a perfomance history from a file."""
    reference_path = Path(__file__).parent / "reference_history.json"
    history = PerformanceHistory.from_file(reference_path)
    reference = PerformanceHistory([-2.0, -3.0], [1.0, 0.0])
    assert history.history_items == reference.history_items
