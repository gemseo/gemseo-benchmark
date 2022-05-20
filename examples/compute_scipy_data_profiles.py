"""Compute data profiles for SciPy algorithms."""
from __future__ import annotations, print_function

from itertools import product
from typing import Any, Callable

from numpy import array
from numpy.core.multiarray import ndarray
from scipy.optimize import minimize, rosen

from gemseo_benchmark.data_profiles.data_profile import DataProfile
from gemseo_benchmark.data_profiles.targets_generator import TargetsGenerator


def generate_values_history(
        objective: Callable[[Any], float],
        method_name: str,
        start_point: ndarray,
) -> list[float]:
    """Minimize a function with a SciPy method.

    Args:
        objective: The objective function to minimize.
        method_name: The name of a SciPy optimization method.
        start_point: The starting point of the algorithm.

    Returns:
       The successive objective values.
    """
    objective_values = list()

    # Wrap the objective function to save its values
    def wrapped_objective(x):
        value = objective(x)
        objective_values.append(value)
        return value

    # Minimize the objective
    minimize(wrapped_objective, start_point, method=method_name)

    return objective_values


# Set the benchmarking problems
objective = rosen
STARTING_POINTS = [
    array([-2.0, -2.0]),
    array([-2.0, 2.0]),
    array([2.0, -2.0]),
    array([2.0, 2.0]),
    array([0.0, 0.0])
]

# Set the reference algorithms
reference_algos = ["slsqp"]

# Generate the reference data
targets_generator = TargetsGenerator()
reference_histories = list()
for ref_algo, start_point in product(reference_algos, STARTING_POINTS):
    objective_history = generate_values_history(objective, ref_algo, start_point)
    targets_generator.add_history(objective_history)

# Compute the scale of target values
targets_number = 20
targets_values = {
    "Rosenbrock": targets_generator.compute_target_values(targets_number, show=True)
}
print("Target values\n", targets_values["Rosenbrock"])

# Set the algorithms to be compared
methods = [
    'nelder-mead', 'powell', 'cg', 'bfgs', 'l-bfgs-b', 'tnc', 'cobyla', 'slsqp',
    'trust-constr',
]

# Compute and plot data profiles
data_profiles = DataProfile(targets_values)
for method_name, start_point in product(methods, STARTING_POINTS):
    history = generate_values_history(objective, method_name, start_point)
    data_profiles.add_history("Rosenbrock", method_name, history)
data_profiles.plot()
