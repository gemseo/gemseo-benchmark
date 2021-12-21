"""Tools to extract data from an optimization problem."""

from typing import Dict, List

from numpy import absolute, atleast_1d, ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction


def get_dimensions(
        problem,  # type: OptimizationProblem
):  # type: (...) -> Dict[str, int]
    """Return the dimensions of the outputs of the problem functions.

    Args:
        problem: The optimization problem.

    Returns:
        The dimensions of the outputs of the problem functions.
    """
    design_variables = problem.design_space.get_current_x()
    outputs, _ = problem.evaluate_functions(
        design_variables, normalize=False,
        no_db_no_norm=problem.nonproc_objective is not None  # FIXME: unnecessary
    )
    return {name: atleast_1d(value).size for name, value in outputs.items()}


def get_n_unsatisfied_constraints(
        problem,  # type: OptimizationProblem
        design_variables,  # type: ndarray
):  # type: (...) -> int
    """Return the number of unsatisfied scalar constraints.

    Args:
        problem: The optimization problem.
        design_variables: The design variables.

    Returns:
        The number of unsatisfied scalar constraints.
    """
    n_unsatisfied = 0
    values, _ = problem.evaluate_functions(
        design_variables, normalize=False,
        no_db_no_norm=problem.nonproc_objective is not None  # FIXME: unnecessary
    )
    for constraint in problem.constraints:
        value = atleast_1d(values[constraint.name])
        if constraint.f_type == MDOFunction.TYPE_EQ:
            value = absolute(value)
            tolerance = problem.eq_tolerance
        else:
            tolerance = problem.ineq_tolerance
        n_unsatisfied += sum(value > tolerance)
    return n_unsatisfied


def get_scalar_constraints_names(
        problem,  # type: OptimizationProblem
):  # type: (...) -> List[str]
    """Get the names of the scalar constraints.

    Args:
        problem: The optimization problem.

    Returns:
        The names of the scalar constraints.
    """
    constraints_names = list()
    dimensions = get_dimensions(problem)
    for name in problem.get_constraints_names():
        dimension = dimensions[name]
        if dimension == 1:
            constraints_names.append(name)
        else:
            constraints_names.extend([
                "{}{}{}".format(name, DesignSpace.SEP, index)
                for index in range(dimension)
            ])
    return constraints_names
