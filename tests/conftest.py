"""Fixtures for the tests."""
from typing import Dict, Union

import pytest
from numpy import array, ndarray

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.utils.py23_compat import mock

design_variables = array([0.0, 1.0])


@pytest.fixture(scope="package")
def design_space():  # type: (...) -> mock.Mock
    """A design space."""
    design_space = mock.Mock()
    design_space.dimension = 2
    design_space.variables_names = ["x"]
    design_space.variables_sizes = {"x": 2}
    design_space.get_current_x = mock.Mock(return_value=design_variables)
    design_space.set_current_x = mock.Mock()
    design_space.unnormalize_vect = lambda _: _
    design_space.untransform_vect = lambda _: _
    return design_space


@pytest.fixture(scope="package")
def objective():  # type: (...) -> mock.Mock
    """An objective function."""
    objective = mock.Mock()
    objective.name = "f"
    return objective


@pytest.fixture(scope="package")
def inequality_constraint():  # type: (...) -> mock.Mock
    """An inequality constraint."""
    ineq_constr = mock.Mock()
    ineq_constr.name = "g"
    ineq_constr.f_type = "ineq"
    return ineq_constr


@pytest.fixture(scope="package")
def equality_constraint():  # type: (...) -> mock.Mock
    """An equality constraint."""
    eq_constr = mock.Mock()
    eq_constr.name = "h"
    eq_constr.f_type = "eq"
    return eq_constr


@pytest.fixture(scope="package")
def functions_values(objective, inequality_constraint, equality_constraint):
    # type: (...) -> Dict[str, Union[float, ndarray]]
    return {
        objective.name: 2.0,
        inequality_constraint.name: array([1.0]),
        equality_constraint.name: array([0.0])
    }


@pytest.fixture(scope="package")
def database(functions_values):  # type: (...) -> mock.Mock
    """A database."""
    database = mock.Mock()
    hashable_array = mock.Mock()
    hashable_array.unwrap = mock.Mock(return_value=design_variables)
    database.items = mock.Mock(return_value=[(hashable_array, functions_values)])
    database.get = mock.Mock(return_value=functions_values)
    database.__len__ = mock.Mock(return_value=1)
    return database


@pytest.fixture(scope="package")
def problem(
        design_space, objective, inequality_constraint, equality_constraint,
        functions_values, database
):  # type: (...) -> mock.Mock
    """A solved optimization problem."""
    problem = mock.Mock(spec=OptimizationProblem)
    problem.ineq_tolerance = 1e-4
    problem.eq_tolerance = 1e-2
    problem.design_space = design_space
    problem.dimension = design_space.dimension
    problem.objective = objective
    problem.nonproc_objective = None
    problem.constraints = [inequality_constraint, equality_constraint]
    problem.get_constraints_names = mock.Mock(
        return_value=[inequality_constraint.name, equality_constraint.name]
    )
    problem.evaluate_functions = mock.Mock(return_value=(functions_values, None))
    problem.database = database
    problem.get_violation_criteria = mock.Mock(return_value=(False, 1.0))
    problem.get_optimum = mock.Mock(return_value=(
        functions_values[objective.name], design_variables, True, functions_values, None
    ))
    problem.minimize_objective = True
    return problem
