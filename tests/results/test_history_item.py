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
"""Tests for the performance history item."""

from __future__ import annotations

import datetime
import re

import pytest

from gemseo_benchmark.results.history_item import HistoryItem


def test_nonnegative_infeasibility_measure():
    """Check the non-negative infeasibility measure exception."""
    with pytest.raises(
        ValueError, match=re.escape("The infeasibility measure is negative: -1.0.")
    ):
        HistoryItem(1.0, -1.0)


def test_eq():
    """Check the equality of history items."""
    assert HistoryItem(1.0, 2.0) == HistoryItem(1.0, 2.0)
    assert HistoryItem(1.0, 2.0) != HistoryItem(2.0, 1.0)


def test_lt():
    """Check the lower inequality of history items."""
    assert HistoryItem(0.0, 2.0) < HistoryItem(1.0, 2.0)
    assert HistoryItem(0.0, 1.0) < HistoryItem(0.0, 2.0)
    assert not HistoryItem(0.0, 2.0) < HistoryItem(1.0, 1.0)


def test_le():
    """Check the lower inequality or equality of history items."""
    assert HistoryItem(1.0, 2.0) <= HistoryItem(1.0, 2.0)
    assert HistoryItem(0.0, 2.0) <= HistoryItem(1.0, 2.0)
    assert HistoryItem(0.0, 1.0) <= HistoryItem(0.0, 2.0)
    assert not HistoryItem(0.0, 2.0) <= HistoryItem(1.0, 1.0)


def test_repr():
    """Check the representation of a history item."""
    assert repr(HistoryItem(1.0, 2.0)) == "(1.0, 2.0)"


def test_unsatisfied_constraints_number():
    """Check the setting of a negative number of unsatisfied constraints."""
    with pytest.raises(
        ValueError,
        match=re.escape("The number of unsatisfied constraints is negative: -1."),
    ):
        HistoryItem(1.0, 1.0, -1)


@pytest.mark.parametrize(("measure", "number"), [(1.0, 0), (0.0, 1)])
def test_inconsistent_unsatisfied_constraints_number(measure, number):
    """Check the setting of an inconsistent number of unsatisfied constraints."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The infeasibility measure ({measure}) and the number of unsatisfied "
            f"constraints ({number}) are not consistent."
        ),
    ):
        HistoryItem(1.0, measure, number)


@pytest.mark.parametrize(("measure", "number"), [(1.0, None), (0.0, 0)])
def test_default_unsatisfied_constraints_number(measure, number):
    """Check the default number of unsatisfied constraints."""
    assert HistoryItem(1.0, measure).n_unsatisfied_constraints == number


def test_copy() -> None:
    """Check the copy of a history item."""
    item = HistoryItem(1, 2, 3)
    copy = item.copy()
    assert copy == item
    assert copy.n_unsatisfied_constraints == item.n_unsatisfied_constraints
    assert copy.elapsed_time == item.elapsed_time
    assert copy.number_of_discipline_executions == item.number_of_discipline_executions
    assert copy is not item


def test_switch_performance_measure_sign() -> None:
    """Check the switch of sign of the performance measure."""
    item = HistoryItem(1, 2, 3)
    item.switch_performance_measure_sign()
    assert item.performance_measure == -1
    assert item.infeasibility_measure == 2
    assert item.n_unsatisfied_constraints == 3


@pytest.mark.parametrize("n_unsatisfied_constraints", [None, 3])
def test_to_dict(n_unsatisfied_constraints) -> None:
    """Check the export to dictionary."""
    reference = {
        "performance measure": 1,
        "infeasibility measure": 2,
        "number of discipline executions": 0,
    }
    if n_unsatisfied_constraints is not None:
        reference["number of unsatisfied constraints"] = n_unsatisfied_constraints

    reference["elapsed time"] = 0
    assert HistoryItem(1, 2, n_unsatisfied_constraints).to_dict() == reference


@pytest.mark.parametrize("n_unsatisfied_constraints", [None, 3])
@pytest.mark.parametrize("elapsed_time", [None, 10])
def test_from_dict(n_unsatisfied_constraints, elapsed_time) -> None:
    """Check the creation from a dictionary."""
    data = {"performance measure": 1, "infeasibility measure": 2}
    if n_unsatisfied_constraints is not None:
        data["number of unsatisfied constraints"] = n_unsatisfied_constraints

    if elapsed_time is not None:
        data["elapsed time"] = elapsed_time

    item = HistoryItem.from_dict(data)
    assert item.performance_measure == 1
    assert item.infeasibility_measure == 2
    assert item.n_unsatisfied_constraints == n_unsatisfied_constraints
    if elapsed_time is None:
        assert item.elapsed_time.total_seconds() == 0
    else:
        assert item.elapsed_time.total_seconds() == elapsed_time


def test_set_elapsed_time() -> None:
    """Check the setting of the elapsed time."""
    item = HistoryItem(1, 2, 3)
    assert item.elapsed_time == datetime.timedelta(seconds=0)
    item.elapsed_time = datetime.timedelta(seconds=10)
    assert item.elapsed_time == datetime.timedelta(seconds=10)


def test_set_number_of_discipline_executions() -> None:
    """Check the setting of the number of discipline executions."""
    item = HistoryItem(1, 2, 3)
    assert item.number_of_discipline_executions == 0
    item.number_of_discipline_executions = 1
    assert item.number_of_discipline_executions == 1
