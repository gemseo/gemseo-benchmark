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
"""Tests of getting data for a plot axis."""

import matplotlib
import matplotlib.pyplot
import pytest
from numpy.testing import assert_equal

from gemseo_benchmark.report.axis_data import ConstraintData
from gemseo_benchmark.report.axis_data import DisciplineData
from gemseo_benchmark.report.axis_data import InfeasibilityData
from gemseo_benchmark.report.axis_data import IterationData
from gemseo_benchmark.report.axis_data import PerformanceData
from gemseo_benchmark.report.axis_data import TimeAbscissaData
from gemseo_benchmark.report.axis_data import TimeOrdinateData
from tests.conftest import check_spread_over_numbers_of_discipline_executions
from tests.conftest import check_spread_over_time


@pytest.fixture
def axes() -> matplotlib.axes.Axes:
    """Axes for plotting."""
    return matplotlib.pyplot.subplots()[1]


def test_performance_data(axes, timed_performance_histories) -> None:
    """Check performance axis data."""
    data = PerformanceData(axes, 10, "Performance measure")
    assert data._label == "Performance measure"
    assert_equal(data.get(timed_performance_histories), [[1, 1], [2, 2]])


def test_infeasibility_data(axes, timed_performance_histories) -> None:
    """Check infeasibility axis data."""
    data = InfeasibilityData(axes)
    assert data._label == "Infeasibility measure"
    assert_equal(data.get(timed_performance_histories), [[0.0, 0.0], [0.0, 0.0]])


def test_constraint_data(axes, timed_performance_histories) -> None:
    """Check constraint axis data."""
    data = ConstraintData(axes)
    assert data._label == "Number of unsatisfied constraints"
    assert_equal(data.get(timed_performance_histories), [[0, 0], [0, 0]])
    assert isinstance(axes.yaxis.get_major_locator(), matplotlib.ticker.MaxNLocator)


def test_time_ordinate_data(axes, timed_performance_histories) -> None:
    """Check time ordinate axis data."""
    data = TimeOrdinateData(axes)
    assert data._label == "Elapsed time"
    assert_equal(data.get(timed_performance_histories), [[1, 3], [2, 4]])
    assert axes.yaxis.get_major_formatter() == data.time_tick_formatter


use_log_scale = pytest.mark.parametrize("use_log_scale", [False, True])


@use_log_scale
def test_discipline_data(axes, multidisciplinary_histories, use_log_scale) -> None:
    """Check discipline axis data."""
    data = DisciplineData(axes, 5, use_log_scale)
    assert data._label == "Number of discipline executions"
    assert_equal(data.get(multidisciplinary_histories), [1, 2, 3, 4])
    if use_log_scale:
        assert axes.get_xscale() == matplotlib.scale.LogScale.name
    else:
        assert isinstance(axes.yaxis.get_major_locator(), matplotlib.ticker.MaxNLocator)

    check_spread_over_numbers_of_discipline_executions(
        data.spread(multidisciplinary_histories)
    )


@use_log_scale
def test_iteration_data(axes, timed_performance_histories, use_log_scale) -> None:
    """Check iteration axis data."""
    data = IterationData(axes, 5, use_log_scale)
    assert data._label == "Number of iterations"
    assert_equal(data.get(timed_performance_histories), [1, 2])
    if use_log_scale:
        assert axes.get_xscale() == matplotlib.scale.LogScale.name
    else:
        assert isinstance(axes.yaxis.get_major_locator(), matplotlib.ticker.MaxNLocator)


@use_log_scale
def test_time_abscissa_data(axes, timed_performance_histories, use_log_scale) -> None:
    """Check time abscissa axis data."""
    data = TimeAbscissaData(axes, 5, use_log_scale)
    assert data._label == "Elapsed time"
    assert_equal(data.get(timed_performance_histories), [1, 2, 3, 4])
    if use_log_scale:
        assert axes.get_xscale() == matplotlib.scale.LogScale.name
    else:
        assert axes.xaxis.get_major_formatter() == data.time_tick_formatter

    check_spread_over_time(data.spread(timed_performance_histories))
