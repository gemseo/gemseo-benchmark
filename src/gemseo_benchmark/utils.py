from typing import Dict, List

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.function import MDOFunction
from numpy import absolute, atleast_1d, ndarray


def get_dimensions(problem):  # type: (...) -> Dict[str, int]
    """Return the dimensions of the outputs of the problem functions.

    Returns:
        The dimensions of the outputs of the problem functions.
    """
    x_vect = problem.design_space.get_current_x()
    outputs, _ = problem.evaluate_functions(x_vect, normalize=False)
    dimensions = dict()
    for name, value in outputs.items():
        value = atleast_1d(value)
        assert value.ndim == 1  # FIXME: get rid off this test
        dimension, = value.shape
        dimensions[name] = dimension
    return dimensions


def get_n_unsatisfied_constraints(
        problem,
        x_vect,  # type: ndarray
):  # type: (...) -> int
    """Return the number of scalar unsatisfied constraints.

    Args:
        x_vect: The design parameters.

    Returns:
        The number of scalar unsatisfied constraints.
    """
    n_unsatisfied = 0
    values = problem.database.get(x_vect)
    for constraint in problem.get_ineq_constraints() + problem.get_eq_constraints():
        value = atleast_1d(values[constraint.name])
        if constraint.f_type == MDOFunction.TYPE_EQ:
            n_unsatisfied += sum(absolute(value) > problem.eq_tolerance)
        else:
            n_unsatisfied += sum(value > problem.ineq_tolerance)
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
