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
from results.results import Results


def test_add_invalid_path():
    """Check the addition of a nonexistent path to a collection."""
    results = Results()
    path = Path(__file__).parent / "not_a_file.json"
    with pytest.raises(FileNotFoundError):
        results.add_path("algo", "problem", path)


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
    with pytest.raises(FileNotFoundError):
        results.from_file(Path("not_a_path.json"))

