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
"""Tests for the performance history item."""
from gemseo_benchmark.results.history_item import HistoryItem
from pytest import raises


def test_nonnegative_infeasibility_measure():
    """Check the non-negative infeasibility measure exception."""
    with raises(ValueError):
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
