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

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import numpy
import pytest

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.results.history_item import HistoryItem
from gemseo_benchmark.results.performance_history import PerformanceHistory


def test_invalid_init_lengths():
    """Check the initialization of a history with lists of inconsistent lengths."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The objective history and the infeasibility history must have same length."
        ),
    ):
        PerformanceHistory([3.0, 2.0], [1.0])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The objective history and the feasibility history must have same length."
        ),
    ):
        PerformanceHistory([3.0, 2.0], feasibility_statuses=[False])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The unsatisfied constraints history and the feasibility history"
            " must have same length."
        ),
    ):
        PerformanceHistory([3.0, 2.0], [1.0, 0.0], n_unsatisfied_constraints=[1])


def test_negative_infeasibility_measures():
    """Check the initialization of a history with negative infeasibility measures."""
    with pytest.raises(ValueError):
        PerformanceHistory([3.0, 2.0], [1.0, -1.0])


def test_length():
    """Check the length of a performance history."""
    history_1 = PerformanceHistory([3.0, 2.0])
    assert len(history_1) == 2
    history_2 = PerformanceHistory([3.0, 2.0], [1.0, 0.0])
    assert len(history_2) == 2
    history_3 = PerformanceHistory([3.0, 2.0], feasibility_statuses=[False, True])
    assert len(history_3) == 2


def test_iter():
    """Check the iteration over a performance history."""
    history = PerformanceHistory([3.0, 2.0], [1.0, 0.0])
    assert list(iter(history)) == [HistoryItem(3.0, 1.0), HistoryItem(2.0, 0.0)]
    history = PerformanceHistory([3.0, 2.0], feasibility_statuses=[False, True])
    assert list(iter(history)) == [HistoryItem(3.0, numpy.inf), HistoryItem(2.0, 0.0)]


@pytest.fixture(scope="module")
def performance_history(algorithm_configuration) -> PerformanceHistory:
    """A performance history."""
    return PerformanceHistory(
        [0.0, -3.0, -1.0, 0.0, 1.0, -1.0],
        [2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        [False, False, False, True, True, True],
        [4, 6, 2, 0, 0, 0],
        "problem",
        "objective",
        ["constraint1", "constraint2"],
        6,
        123,
        algorithm_configuration,
        2,
    )


def check_non_item_attributes(history, reference_history):
    """Check the non-item attributes of a performance history against a reference."""
    assert history.problem_name == reference_history.problem_name
    assert history._objective_name == reference_history._objective_name
    assert history._constraints_names == reference_history._constraints_names
    assert history.doe_size == reference_history.doe_size
    assert history.total_time == reference_history.total_time
    assert history.algorithm_configuration == reference_history.algorithm_configuration
    assert history._number_of_variables == reference_history._number_of_variables


def test_compute_cumulated_minimum(performance_history, algorithm_configuration):
    """Check the computation of the cumulated minimum of a performance history."""
    PerformanceHistory([0.0, 0.0, -1.0, 0.0, 0.0, -1.0], [2.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    cumulated_minimum = performance_history.compute_cumulated_minimum()
    check_non_item_attributes(cumulated_minimum, performance_history)
    assert cumulated_minimum.items == [
        HistoryItem(0, 2, 4),
        HistoryItem(0, 2, 4),
        HistoryItem(-1, 1, 2),
        HistoryItem(0, 0, 0),
        HistoryItem(0, 0, 0),
        HistoryItem(-1, 0, 0),
    ]


history_1 = PerformanceHistory([1.0, -1.0, 0.0], [2.0, 0.0, 3.0])
history_2 = PerformanceHistory([-2.0, -2.0, 2.0], [0.0, 3.0, 0.0])
history_3 = PerformanceHistory([3.0, -3.0, 3.0], [0.0, 0.0, 0.0])


def test_compute_minimum_history():
    """Check the computation of the minimum history."""
    items = [HistoryItem(-2.0, 0.0), HistoryItem(-3.0, 0.0), HistoryItem(2.0, 0.0)]
    minimum = PerformanceHistory.compute_minimum_history([
        history_1,
        history_2,
        history_3,
    ])
    assert minimum.items == items


def test_compute_maximum_history():
    """Check the computation of the maximum history."""
    items = [HistoryItem(1.0, 2.0), HistoryItem(-2.0, 3.0), HistoryItem(0.0, 3.0)]
    maximum = PerformanceHistory.compute_maximum_history([
        history_1,
        history_2,
        history_3,
    ])
    assert maximum.items == items


def test_compute_median_history():
    """Check the computation of the median history."""
    items = [HistoryItem(3.0, 0.0), HistoryItem(-1.0, 0.0), HistoryItem(3.0, 0.0)]
    median = PerformanceHistory.compute_median_history([
        history_1,
        history_2,
        history_3,
    ])
    assert median.items == items


def test_remove_leading_infeasible(performance_history):
    """Check the removal of the leading infeasible items in a performance history."""
    truncation = performance_history.remove_leading_infeasible()
    check_non_item_attributes(truncation, performance_history)
    assert truncation.items == [
        HistoryItem(0, 0, 0),
        HistoryItem(1, 0, 0),
        HistoryItem(-1, 0, 0),
    ]


def test_remove_leading_infeasible_from_infeasible_history() -> None:
    """Check the removal of the leading infeasible items from an infeasible history."""
    performance_history = PerformanceHistory([3, 2, 1], [1, 1, 1])
    truncation = performance_history.remove_leading_infeasible()
    check_non_item_attributes(truncation, performance_history)
    assert truncation.items == []


def test_to_file(tmp_path):
    """Check the writing of a performance history into a file."""
    algorithm_configuration = AlgorithmConfiguration(
        "algorithm", optional_path=Path("path")
    )
    history = PerformanceHistory(
        [-2.0, -3.0],
        [1.0, 0.0],
        n_unsatisfied_constraints=[1, 0],
        problem_name="problem",
        objective_name="f",
        constraints_names=["g", "h"],
        doe_size=7,
        total_time=123.45,
        algorithm_configuration=algorithm_configuration,
        number_of_variables=4,
    )
    file_path = tmp_path / "history.json"
    history.to_file(file_path)
    with file_path.open("r") as file:
        contents = file.read()

    reference_path = Path(__file__).parent / "reference_history.json"
    with reference_path.open("r") as reference_file:
        reference = reference_file.read()

    assert contents == reference[:-1]  # disregard last line break


def test_from_file():
    """Check the initialization of a performance history from a file."""
    reference_path = Path(__file__).parent / "reference_history.json"
    history = PerformanceHistory.from_file(reference_path)
    assert history.problem_name == "problem"
    assert history._number_of_variables == 4
    assert history._objective_name == "f"
    assert history._constraints_names == ["g", "h"]
    assert history.algorithm_configuration.algorithm_name == "algorithm"
    assert history.algorithm_configuration.name == "algorithm_optional_path='path'"
    assert history.algorithm_configuration.algorithm_options == {
        "optional_path": "path"
    }
    assert history.doe_size == 7
    assert history.total_time == 123.45
    assert history.items[0].objective_value == -2.0
    assert history.items[0].infeasibility_measure == 1.0
    assert history.items[0].n_unsatisfied_constraints == 1
    assert history.items[1].objective_value == -3.0
    assert history.items[1].infeasibility_measure == 0.0
    assert history.items[1].n_unsatisfied_constraints == 0


def test_history_items_setter():
    """Check the setting of history items."""
    history = PerformanceHistory()
    with pytest.raises(
        TypeError, match=re.escape("History items must be of type HistoryItem.")
    ):
        history.items = [1.0, 2.0]


def test_repr():
    """Check the representation of a performance history."""
    history = PerformanceHistory([-2.0, -3.0], [1.0, 0.0])
    assert repr(history) == "[(-2.0, 1.0), (-3.0, 0.0)]"


def test_from_problem(problem, database):
    """Check the creation of a performance history out of a solved problem."""
    problem.database = database
    history = PerformanceHistory.from_problem(problem, "problem")
    assert history.objective_values == [2.0]
    assert history.infeasibility_measures == [1.0]
    assert history.n_unsatisfied_constraints == [1]


@pytest.fixture(scope="module")
def incomplete_database(objective, equality_constraint, hashable_array) -> mock.Mock:
    """An incomplete database."""
    database = mock.Mock()
    functions_values = {
        objective.name: 2.0,
        equality_constraint.name: numpy.array([0.0]),
    }
    database.items = mock.Mock(return_value=[(hashable_array, functions_values)])
    # database.get = mock.Mock(return_value=functions_values)  # FIXME
    # database.__len__ = mock.Mock(return_value=1)  # FIXME
    return database


def test_from_problem_incomplete_database(problem, incomplete_database):
    """Check the creation of a performance history out of an incomplete database."""
    problem.database = incomplete_database
    history = PerformanceHistory.from_problem(problem, "problem")
    assert len(history) == 0


@pytest.mark.parametrize("size", [6, 9])
def test_extend(performance_history, size):
    """Check the extension of a performance history."""
    extension = performance_history.extend(size)
    check_non_item_attributes(extension, performance_history)
    assert len(extension) == size
    assert extension.items[:6] == performance_history.items[:6]
    assert extension[size - 1] == performance_history[5]


def test_extend_smaller(performance_history):
    """Check the extension of a performance history to a smaller size."""
    with pytest.raises(
        ValueError,
        match=re.escape("The expected size (1) is smaller than the history size (6)."),
    ):
        performance_history.extend(1)


@pytest.mark.parametrize("size", [1, 6])
def test_shorten(performance_history, size):
    """Check the shortening of a performance history."""
    shortening = performance_history.shorten(size)
    check_non_item_attributes(shortening, performance_history)
    assert len(shortening) == size
    assert shortening.items == performance_history.items[:size]


def test_get_plot_data_feasible():
    """Check the retrieval of feasible data for plotting."""
    history = PerformanceHistory([2.0, 1.0], [1.0, 1.0])
    assert history.get_plot_data(feasible=True) == ([], [])
