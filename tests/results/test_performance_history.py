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
"""Tests for the performance history."""
import json

import pytest
from gemseo.utils.py23_compat import mock, Path
from gemseo_benchmark.results.history_item import HistoryItem
from gemseo_benchmark.results.performance_history import PerformanceHistory
from numpy import array, inf
from pytest import raises


def test_invalid_init_lengths():
    """Check the initialization of a history with lists of inconsistent lengths."""
    with raises(
            ValueError,
            match="The objective history and the infeasibility history must have same"
                  " length."
    ):
        PerformanceHistory([3.0, 2.0], [1.0])
    with raises(
            ValueError,
            match="The objective history and the feasibility history must have same"
                  " length."
    ):
        PerformanceHistory([3.0, 2.0], feasibility_statuses=[False])
    with pytest.raises(
            ValueError,
            match="The unsatisfied constraints history and the feasibility history"
                  " must have same length."
    ):
        PerformanceHistory([3.0, 2.0], [1.0, 0.0], n_unsatisfied_constraints=[1])


def test_negative_infeasibility_measures():
    """Check the initialization of a history with negative infeasibility measures."""
    with raises(ValueError):
        PerformanceHistory([3.0, 2.0], [1.0, -1.0])


def test_length():
    """Check the length of a performance history"""
    history_1 = PerformanceHistory([3.0, 2.0])
    assert len(history_1) == 2
    history_2 = PerformanceHistory([3.0, 2.0], [1.0, 0.0])
    assert len(history_2) == 2
    history_3 = PerformanceHistory([3.0, 2.0], feasibility_statuses=[False, True])
    assert len(history_3) == 2


def test_iter():
    """Check the iteration over a performance history"""
    history = PerformanceHistory([3.0, 2.0], [1.0, 0.0])
    assert list(iter(history)) == [HistoryItem(3.0, 1.0), HistoryItem(2.0, 0.0)]
    history = PerformanceHistory([3.0, 2.0], feasibility_statuses=[False, True])
    assert list(iter(history)) == [HistoryItem(3.0, inf), HistoryItem(2.0, 0.0)]


def test_compute_cumulated_minimum():
    """Check the computation of the cumulated minimum of a performance history"""
    history = PerformanceHistory(
        [0.0, -3.0, -1.0, 0.0, 1.0, -1.0],
        [2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    )
    reference = PerformanceHistory(
        [0.0, 0.0, -1.0, 0.0, 0.0, -1.0], [2.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    )
    cumulated_minimum = history.compute_cumulated_minimum()
    assert cumulated_minimum.history_items == reference.history_items


def test_compute_median_history():
    """Check the computation of the median history."""
    hist_1 = PerformanceHistory([1.0, -1.0, 0.0], [2.0, 0.0, 3.0])
    hist_2 = PerformanceHistory([-2.0, -2.0, 2.0], [0.0, 3.0, 0.0])
    hist_3 = PerformanceHistory([3.0, -3.0, 3.0], [0.0, 0.0, 0.0])
    reference = PerformanceHistory([3.0, -1.0, 3.0], [0.0, 0.0, 0.0])
    median = PerformanceHistory.compute_median_history([hist_1, hist_2, hist_3])
    assert median.history_items == reference.history_items


def test_remove_leading_infeasible():
    """Check the removal of the leading infeasible items in a performance history."""
    history = PerformanceHistory([2.0, 1.0, 0.0, 1.0, -1.0], [2.0, 1.0, 0.0, 3.0, 0.0])
    reference = PerformanceHistory([0.0, 1.0, -1.0], [0.0, 3.0, 0.0])
    truncation = history.remove_leading_infeasible()
    assert truncation.history_items == reference.history_items


def test_to_file(tmp_path):
    """Check the writing of a performance history into a file."""
    history = PerformanceHistory(
        [-2.0, -3.0], [1.0, 0.0], n_unsatisfied_constraints=[1, 0]
    )
    file_path = tmp_path / "history.json"
    history.to_file(str(file_path))
    with file_path.open("r") as file:
        contents = file.read()
    reference_path = Path(__file__).parent / "reference_history.json"
    with reference_path.open("r") as reference_file:
        reference = reference_file.read()
    assert contents == reference


def test_from_file():
    """Check the initialization of a perfomance history from a file."""
    reference_path = Path(__file__).parent / "reference_history.json"
    history = PerformanceHistory.from_file(reference_path)
    reference = PerformanceHistory([-2.0, -3.0], [1.0, 0.0])
    assert history.history_items == reference.history_items


def test_history_items_setter():
    """Check the setting of history items."""
    history = PerformanceHistory()
    with raises(TypeError, match="History items must be of type HistoryItem"):
        history.history_items = [1.0, 2.0]


def test_repr():
    """Check the representation of a performance history."""
    history = PerformanceHistory([-2.0, -3.0], [1.0, 0.0])
    assert repr(history) == "[(-2.0, 1.0), (-3.0, 0.0)]"


def test_to_postpro_json(tmp_path):
    """Check the saving to a JSON a file."""
    objective_values = [0.0, -3.0, -1.0, 0.0, 1.0, -1.0]
    infeasibility_measures = [2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    problem_name = "problem"
    objective_name = "f"
    constraints_names = ["g", "h"]
    doe_size = 10
    nbr_eval_iter = 1
    n_unsatisfied_constraints = [1, 1, 1, 0, 0, 0]
    population_size = 1
    total_time = 1.0
    algorithm = "algo"
    history = PerformanceHistory(
        objective_values, infeasibility_measures,
        n_unsatisfied_constraints=n_unsatisfied_constraints, problem_name=problem_name,
        objective_name=objective_name, constraints_names=constraints_names,
        doe_size=doe_size, nbr_eval_iter=nbr_eval_iter, population_size=population_size,
        total_time=total_time, algorithm=algorithm
    )
    path = tmp_path / "history_postpro.json"
    path = Path(__file__).parent / path.name
    history.to_postpro_json(path)
    # Check the output JSON file
    with path.open("r") as file:
        contents = json.load(file)
    assert isinstance(contents, dict)
    assert contents["version"] == algorithm
    assert contents["responses"] == ["f", "g", "h"]
    assert contents["objective"] == [0.0, 0.0, -1.0, 0.0, 0.0, -1.0]
    assert contents["doe_size"] == doe_size
    assert contents["nbr_eval_iter"] == nbr_eval_iter
    assert contents["num_const"] == n_unsatisfied_constraints
    assert contents["population"] == population_size
    assert contents["name"] == problem_name
    assert contents["total_time"] == total_time


def test_to_postpro_no_algo(tmp_path):
    """Check the export to post-processing JSON when no algorithm is set."""
    with pytest.raises(ValueError, match="The algorithm name is not set."):
        PerformanceHistory().to_postpro_json(tmp_path)


@pytest.fixture(scope="module")
def objective():  # type: (...) -> mock.Mock
    """An objective constraint."""
    objective = mock.Mock()
    objective.name = "f"
    return objective


@pytest.fixture(scope="module")
def ineq_constr():  # type: (...) -> mock.Mock
    """An inequality constraint."""
    ineq_constr = mock.Mock()
    ineq_constr.name = "g"
    ineq_constr.f_type = "ineq"
    return ineq_constr


@pytest.fixture(scope="module")
def eq_constr():  # type: (...) -> mock.Mock
    """An equality constraint."""
    eq_constr = mock.Mock()
    eq_constr.name = "h"
    eq_constr.f_type = "eq"
    return eq_constr


@pytest.fixture(scope="module")
def problem(objective, ineq_constr, eq_constr):  # type: (...) -> mock.Mock
    """A solved optimization problem."""
    x_vect = array([0.0, 1.0])
    values = {
        objective.name: 2.0,
        ineq_constr.name: array([1.0]),
        eq_constr.name: array([0.0])
    }
    problem = mock.Mock()
    problem.ineq_tolerance = 1e-4
    problem.eq_tolerance = 1e-2
    problem.design_space.get_current_x = mock.Mock(return_value=x_vect)
    problem.objective = objective
    problem.get_ineq_constraints = mock.Mock(return_value=[ineq_constr])
    problem.get_eq_constraints = mock.Mock(return_value=[eq_constr])
    problem.get_constraints_names = mock.Mock(
        return_value=[ineq_constr.name, eq_constr.name]
    )
    problem.evaluate_functions = mock.Mock(return_value=(values, None))
    problem.database.items = mock.Mock(return_value=[(x_vect, values)])
    problem.database.get = mock.Mock(return_value=values)
    problem.get_violation_criteria = mock.Mock(return_value=(False, 1.0))
    return problem


def test_from_problem(problem):
    """Check the creation of a performance history out of a solved problem."""
    history = PerformanceHistory.from_problem(problem, "problem")
    assert history.objective_values == [2.0]
    assert history.infeasibility_measures == [1.0]
    assert history.n_unsatisfied_constraints == [1]
