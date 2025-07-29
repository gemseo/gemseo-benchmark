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
from datetime import timedelta
from math import isnan
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
            "The performance history and the infeasibility history "
            "must have same length: 2 != 1."
        ),
    ):
        PerformanceHistory([3.0, 2.0], [1.0])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The performance history and the feasibility history must have same length:"
            " 2 != 1."
        ),
    ):
        PerformanceHistory([3.0, 2.0], feasibility_statuses=[False])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The unsatisfied constraints history and the infeasibility history"
            " must have same length: 1 != 2."
        ),
    ):
        PerformanceHistory([3.0, 2.0], [1.0, 0.0], n_unsatisfied_constraints=[1])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The performance history and the elasped time history "
            "must have same length: 2 != 1."
        ),
    ):
        PerformanceHistory([3.0, 2.0], elapsed_times=[timedelta(seconds=10)])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The performance history "
            "and the number of discipline executions history "
            "must have same length: 2 != 1."
        ),
    ):
        PerformanceHistory([3.0, 2.0], number_of_discipline_executions=[10])


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


def test_compute_cumulated_minimum(performance_history) -> None:
    """Check the computation of the cumulated minimum of a performance history."""
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
    assert cumulated_minimum[0] is not cumulated_minimum[1]
    assert cumulated_minimum[3] is not cumulated_minimum[4]


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


def test_to_file_empty(tmp_path):
    """Check the writing of an empty performance history into a file."""
    file_path = tmp_path / "history.json"
    PerformanceHistory().to_file(file_path)
    with file_path.open("r") as file:
        assert (
            file.read()
            == """{
  "history_items": []
}"""
        )


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
    assert history.items[0].performance_measure == -2.0
    assert history.items[0].infeasibility_measure == 1.0
    assert history.items[0].n_unsatisfied_constraints == 1
    assert history.items[1].performance_measure == -3.0
    assert history.items[1].infeasibility_measure == 0.0
    assert history.items[1].n_unsatisfied_constraints == 0


def test_repr():
    """Check the representation of a performance history."""
    history = PerformanceHistory([-2.0, -3.0], [1.0, 0.0])
    assert repr(history) == "[(-2.0, 1.0), (-3.0, 0.0)]"


def test_from_problem(minimization_problem, database):
    """Check the creation of a performance history out of a solved problem."""
    minimization_problem.database = database
    history = PerformanceHistory.from_problem(minimization_problem, "problem")
    assert history.performance_measures == [2.0]
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


def test_from_problem_incomplete_database(minimization_problem, incomplete_database):
    """Check the creation of a performance history out of an incomplete database."""
    minimization_problem.database = incomplete_database
    history = PerformanceHistory.from_problem(minimization_problem, "problem")
    assert len(history) == 0


@pytest.mark.parametrize("size", [6, 9])
def test_extend(performance_history, size):
    """Check the extension of a performance history."""
    extension = performance_history.extend(size)
    check_non_item_attributes(extension, performance_history)
    assert len(extension) == size
    assert extension.items[:6] == performance_history.items[:6]
    assert extension[size - 1] == performance_history[5]
    if size == 9:
        assert extension[6] is not extension[7]
        assert extension[6] is not extension[8]
        assert extension[7] is not extension[8]


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


def test_switch_performance_measure_sign() -> None:
    """Check the switch of sign of the performance measure."""
    history = PerformanceHistory([1, 2], [3, 4])
    history.switch_performance_measure_sign()
    assert history.performance_measures == [-1, -2]
    assert history.infeasibility_measures == [3, 4]


def test_spread_over_timeline() -> None:
    """Check the spreading of a performance history over a timeline."""
    history = PerformanceHistory(
        [1, 2, 3], elapsed_times=[timedelta(seconds=seconds) for seconds in (2, 4, 6)]
    ).spread_over_timeline([timedelta(seconds=seconds) for seconds in range(1, 10)], 4)
    assert isnan(history[0].performance_measure)
    assert history.n_unsatisfied_constraints == [4] + [0] * 8
    assert history[0].elapsed_time == timedelta(seconds=1)
    assert [
        (item.performance_measure, item.n_unsatisfied_constraints, item.elapsed_time)
        for item in history.items[1:]
    ] == [(1, 0, timedelta(seconds=seconds)) for seconds in range(2, 4)] + [
        (2, 0, timedelta(seconds=seconds)) for seconds in range(4, 6)
    ] + [(3, 0, timedelta(seconds=seconds)) for seconds in range(6, 10)]


def test_spread_over_numbers_of_discipline_executions() -> None:
    """Check the spreading of a performance history over a numbers of executions."""
    history = PerformanceHistory(
        [1, 2, 3], number_of_discipline_executions=[2, 4, 6]
    ).spread_over_numbers_of_discipline_executions(range(1, 10), 4)
    assert isnan(history[0].performance_measure)
    assert history.n_unsatisfied_constraints == [4] + [0] * 8
    assert history[0].number_of_discipline_executions == 1
    assert [
        (
            item.performance_measure,
            item.n_unsatisfied_constraints,
            item.number_of_discipline_executions,
        )
        for item in history.items[1:]
    ] == [(1, 0, executions) for executions in range(2, 4)] + [
        (2, 0, executions) for executions in range(4, 6)
    ] + [(3, 0, executions) for executions in range(6, 10)]
