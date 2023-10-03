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
"""Test for the performance histories collection."""
from __future__ import annotations

import pytest
from gemseo_benchmark.results.history_item import HistoryItem
from gemseo_benchmark.results.performance_histories import PerformanceHistories
from gemseo_benchmark.results.performance_history import PerformanceHistory


@pytest.fixture(scope="module")
def performance_histories() -> PerformanceHistories:
    """A collection of performance histories."""
    return PerformanceHistories(
        PerformanceHistory([1.0, -1.0, 0.0], [2.0, 0.0, 3.0]),
        PerformanceHistory([-2.0, -2.0, 2.0], [0.0, 3.0, 0.0]),
        PerformanceHistory([3.0, -3.0, 3.0], [0.0, 0.0, 0.0]),
        PerformanceHistory([0.0, -2.0, 4.0], [0.0, 0.0, 0.0]),
    )


def test_compute_minimum(performance_histories):
    """Check the computation of the minimum history."""
    assert performance_histories.compute_minimum().items == [
        HistoryItem(-2.0, 0.0),
        HistoryItem(-3.0, 0.0),
        HistoryItem(2.0, 0.0),
    ]


def test_compute_maximum(performance_histories):
    """Check the computation of the maximum history."""
    assert performance_histories.compute_maximum().items == [
        HistoryItem(1.0, 2.0),
        HistoryItem(-2.0, 3.0),
        HistoryItem(0.0, 3.0),
    ]


def test_compute_low_median(performance_histories):
    """Check the computation of the low median history."""
    assert performance_histories.compute_median().items == [
        HistoryItem(0.0, 0.0),
        HistoryItem(-2.0, 0.0),
        HistoryItem(3.0, 0.0),
    ]


def test_compute_high_median(performance_histories):
    """Check the computation of the high median history."""
    assert performance_histories.compute_median(False).items == [
        HistoryItem(3.0, 0.0),
        HistoryItem(-1.0, 0.0),
        HistoryItem(4.0, 0.0),
    ]


def test_set():
    """Check the setting of a performance history."""
    histories = PerformanceHistories(PerformanceHistory(range(3, 6)))
    histories[0] = PerformanceHistory(range(3))
    assert all(
        item == HistoryItem(index, 0.0) for index, item in enumerate(histories[0].items)
    )


def test_del():
    """Check the deletion of a performance history."""
    histories = PerformanceHistories(PerformanceHistory(range(3)))
    del histories[0]
    with pytest.raises(IndexError, match="list index out of range"):
        histories[0]
