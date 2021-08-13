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
"""Tests for the collection of paths to performance histories."""
import json

import pytest
from gemseo.utils.py23_compat import Path
from gemseo_benchmark.results.results import Results


def test_add_invalid_path():
    """Check the addition of a nonexistent path to a collection."""
    results = Results()
    with pytest.raises(
            FileNotFoundError,
            match="The path to the history does not exist: not_a_file.json."
    ):
        results.add_path("algo", "problem", "not_a_file.json")


def test_to_file(tmpdir):
    """Check the saving of a collection of paths to performance histories."""
    results = Results()
    history_path = Path(__file__).parent / "history.json"
    results.add_path("algo", "problem", history_path)
    results_path = tmpdir / "results.json"
    results.to_file(results_path)
    with results_path.open("r") as file:
        contents = json.load(file)
    assert contents == {"algo": {"problem": [str(history_path.resolve())]}}


def test_from_file(tmpdir):
    """Check the loading of a collection of paths to performance histories."""
    results = Results()
    results.from_file(Path(__file__).parent / "results.json")
    # Save the results to check their contents as a file
    results_path = tmpdir / "results.json"
    results.to_file(results_path)
    with results_path.open("r") as file:
        contents = json.load(file)
    history_path = Path(__file__).parent / "history.json"
    assert contents == {"algo": {"problem": [str(history_path.resolve())]}}


def test_from_invalid_file():
    """Check the loading of a collection to an invalid path."""
    results = Results()
    with pytest.raises(
            FileNotFoundError,
            match="The path to the JSON file does not exist: not_a_path.json."
    ):
        results.from_file("not_a_path.json")


def test_algorithms():
    """Check the accessor to the algorithms names."""
    results = Results()
    history_path = Path(__file__).parent / "history.json"
    results.add_path("algo", "problem", history_path)
    assert results.algorithms == ["algo"]


def test_get_problems():
    """Check the accessor to the problems names."""
    results = Results()
    history_path = Path(__file__).parent / "history.json"
    results.add_path("algo", "problem", history_path)
    assert results.get_problems("algo") == ["problem"]


def test_test_paths():
    """Check the accessor to the performance histories paths."""
    results = Results()
    history_path = Path(__file__).parent / "history.json"
    results.add_path("algo", "problem", history_path)
    assert results.get_paths("algo", "problem") == [history_path.resolve()]
