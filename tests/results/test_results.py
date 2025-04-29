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

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from gemseo_benchmark.results.results import Results

algorithm_name = "algorithm"
problem_name = "problem"
history_path = Path(__file__).parent / "history.json"


@pytest.fixture(scope="module")
def results() -> Results:
    """A collection of performance histories."""
    results = Results()
    results.add_path(algorithm_name, problem_name, history_path)
    return results


@pytest.fixture(scope="module")
def results_contents() -> dict[str, dict[str, list[str]]]:
    """The paths for the performance histories."""
    return {algorithm_name: {problem_name: [str(history_path.resolve())]}}


@pytest.fixture
def results_file(tmp_path, results_contents) -> Path:
    """The path to the results file."""
    results_path = tmp_path / "results_reference.json"
    with results_path.open("w") as file:
        json.dump(results_contents, file)
    return results_path


def test_init_from_file(results_file):
    """Check the initialization from a file."""
    results = Results(results_file)
    assert results.get_paths(algorithm_name, problem_name) == [history_path]


def test_add_invalid_path():
    """Check the addition of a nonexistent path to a collection."""
    results = Results()
    with pytest.raises(
        FileNotFoundError,
        match=re.escape("The path to the history does not exist: not_a_file.json."),
    ):
        results.add_path(algorithm_name, problem_name, "not_a_file.json")


def test_to_file(tmp_path, results, results_contents):
    """Check the saving of a collection of paths to performance histories."""
    results_path = tmp_path / "results.json"
    results.to_file(results_path)
    with results_path.open("r") as file:
        contents = json.load(file)
    assert contents == results_contents


def test_from_file(tmp_path, results_contents, results_file):
    """Check the loading of a collection of paths to performance histories."""
    results = Results()
    results.from_file(results_file)
    assert results.get_paths(algorithm_name, problem_name) == [history_path]


def test_from_invalid_file():
    """Check the loading of a collection to an invalid path."""
    results = Results()
    with pytest.raises(
        FileNotFoundError,
        match=re.escape("The path to the JSON file does not exist: not_a_path.json."),
    ):
        results.from_file("not_a_path.json")


def test_algorithms(results):
    """Check the accessor to the algorithms names."""
    assert results.algorithms == [algorithm_name]


def test_get_problems(results):
    """Check the accessor to the problems names."""
    assert results.get_problems(algorithm_name) == [problem_name]


def test_get_problems_unknown_algorithm(results):
    """Check the accessor to the problems names for an unknown algorithm."""
    with pytest.raises(ValueError, match=re.escape("Unknown algorithm name: unknown.")):
        results.get_problems("unknown")


def test_get_paths(results):
    """Check the accessor to the performance histories paths."""
    assert results.get_paths(algorithm_name, problem_name) == [history_path.resolve()]


def test_get_paths_unknown_algorithm(results):
    """Check the accessor to the histories paths for an unknown algorithm."""
    with pytest.raises(ValueError, match=re.escape("Unknown algorithm name: unknown.")):
        results.get_paths("unknown", problem_name)


def test_get_paths_unknown_problem(results):
    """Check the accessor to the histories paths for an unknown problem."""
    with pytest.raises(ValueError, match=re.escape("Unknown problem name: unknown.")):
        results.get_paths(algorithm_name, "unknown")


@pytest.mark.parametrize(
    ("algorithm", "problem", "path", "contained"),
    [
        ("unknown", problem_name, history_path, False),
        (algorithm_name, "unknown", history_path, False),
        (algorithm_name, problem_name, Path(__file__).parent / "unknown", False),
        (algorithm_name, problem_name, history_path, True),
    ],
)
def test_contains(results, algorithm, problem, path, contained):
    """Check the membership assessment of a history path to the results."""
    assert results.contains(algorithm, problem, path) == contained


@pytest.mark.parametrize("empty", [False, True])
def test_remove_paths(empty) -> None:
    """Check the removal of paths."""
    results = Results()
    if not empty:
        results.add_path(algorithm_name, problem_name, history_path)
        assert problem_name in results.get_problems(algorithm_name)
        assert results.contains(algorithm_name, problem_name, history_path)

    results.remove_paths(algorithm_name, problem_name)
    if not empty:
        assert problem_name not in results.get_problems(algorithm_name)
        assert not results.contains(algorithm_name, problem_name, history_path)
