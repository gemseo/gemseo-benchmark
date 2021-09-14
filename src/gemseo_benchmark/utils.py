from typing import Dict

from numpy import atleast_1d


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
        assert value.ndim == 1
        dimension, = value.shape
        dimensions[name] = dimension
    return dimensions
