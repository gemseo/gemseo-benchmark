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

import matplotlib
import matplotlib.pyplot
import matplotlib.testing
import pytest
from matplotlib.testing.decorators import image_comparison

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


@pytest.fixture(scope="module")
def five_performance_histories() -> PerformanceHistories:
    """A collection of performance histories."""
    return PerformanceHistories(
        PerformanceHistory(
            [4, 3, 2, 1, 0, -1, -2],
            [10, 8, 6, 4, 2, 0, 0],
            n_unsatisfied_constraints=[5, 4, 3, 2, 1, 0, 0],
        ),
        PerformanceHistory(
            [3, 2, 1, 0, -1, -2, -3],
            [8, 6, 4, 2, 0, 0, 0],
            n_unsatisfied_constraints=[4, 3, 2, 1, 0, 0, 0],
        ),
        PerformanceHistory(
            [2, 1, 0, -1, -2, -3, -4],
            [6, 4, 2, 0, 0, 0, 0],
            n_unsatisfied_constraints=[3, 2, 1, 0, 0, 0, 0],
        ),
        PerformanceHistory(
            [1, 0, -1, -2, -3, -4, -5],
            [4, 2, 0, 0, 0, 0, 0],
            n_unsatisfied_constraints=[2, 1, 0, 0, 0, 0, 0],
        ),
        PerformanceHistory(
            [0, -1, -2, -3, -4, -5, -6],
            [2, 0, 0, 0, 0, 0, 0],
            n_unsatisfied_constraints=[1, 0, 0, 0, 0, 0, 0],
        ),
    )


@image_comparison(["performance_measure_distribution"], ["png"])
def test_plot_performance_measure_distribution(
    five_performance_histories,
):
    """Check the plot of the distribution of the performance measure."""
    matplotlib.pyplot.close("all")
    five_performance_histories.plot_performance_measure_distribution(
        matplotlib.pyplot.figure().gca(), 4
    )


@image_comparison(["performance_measure_distribution_infeasible_centile"], ["png"])
def test_plot_performance_measure_distribution_infeasible_centile():
    """Check the plotting of an infeasible centile."""
    matplotlib.pyplot.close("all")
    PerformanceHistories(
        PerformanceHistory([6, 5, 4], [1, 1, 1]),
        PerformanceHistory([5, 4, 3], [1, 1, 1]),
        PerformanceHistory([4, 3, 2], [1, 1, 1]),
        PerformanceHistory([3, 2, 1], [1, 1, 1]),
        PerformanceHistory([2, 1, 0], [1, 0, 0]),
    ).plot_performance_measure_distribution(matplotlib.pyplot.figure().gca())


@image_comparison(["infeasibility_measure_distribution"], ["png"])
def test_plot_infeasibility_measure_distribution(
    five_performance_histories,
):
    """Check the plot of the distribution of the infeasibility measure."""
    matplotlib.pyplot.close("all")
    five_performance_histories.plot_infeasibility_measure_distribution(
        matplotlib.pyplot.figure().gca()
    )


@image_comparison(["number_of_unsatisfied_constraints_distribution"], ["png"])
def test_plot_number_of_unsatisfied_constraints_distribution(
    five_performance_histories,
):
    """Check the plot of the distribution of the number of unsatisfied constraints."""
    matplotlib.pyplot.close("all")
    five_performance_histories.plot_number_of_unsatisfied_constraints_distribution(
        matplotlib.pyplot.figure().gca()
    )
