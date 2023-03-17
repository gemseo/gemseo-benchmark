"""An example of computation of target values."""
from __future__ import annotations

from gemseo.api import compute_doe
from gemseo.api import configure_logger
from gemseo.problems.analytical.power_2 import Power2
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.problems.problem import Problem

configure_logger()

# Create the benchmarking problem
problem = Problem(
    name="Power2",
    optimization_problem_creator=Power2,
    optimum=2.0 ** (1.0 / 3.0) + 0.9 ** (2.0 / 3.0),
)

# Define 10 starting points from an optimal latin hypercube sampling.
design_space = problem.creator().design_space
problem.start_points = compute_doe(design_space, "OT_OPT_LHS", 10)

# Define the convergence configurations for the optimization algorithms.
algorithms_configurations = AlgorithmsConfigurations(
    AlgorithmConfiguration(
        "NLOPT_COBYLA", max_iter=100, eq_tolerance=1e-4, ineq_tolerance=0.0
    )
)

# Compute five target values for the benchmarking problem.
# This automatic procedure has two stages:
# 1/ execute the specified algorithms once for each of the starting points,
# 2/ automatically select target values based on the algorithms histories.
# These targets represent the milestones of the problem resolution.
problem.compute_targets(5, algorithms_configurations, best_target_tolerance=1e-5)

# Plot the algorithms histories used as reference for the computation of the target
# values, with the objective value on the vertical axis and the number of functions
# evaluations of the horizontal axis.
problem.targets_generator.plot_histories(best_target_value=problem.optimum, show=True)

# Plot the target values: the objective value of each of the 5 targets is represented
# on the vertical axis with a marker indicating whether the target is feasible or not.
problem.target_values.plot()
