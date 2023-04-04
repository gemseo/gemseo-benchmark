"""A simple script to execute a benchmarking scenario."""
from __future__ import annotations

from pathlib import Path

from gemseo.api import configure
from gemseo.problems.analytical.rastrigin import Rastrigin
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.scenario import Scenario

# Deactivate functions counters, progress bars and bounds check to accelerate the script
configure(
    activate_function_counters=False,
    activate_progress_bar=False,
    check_desvars_bounds=False,
)

# Select the algorithms configurations to be benchmarked.
slsqp = AlgorithmConfiguration("SLSQP")
lbfgsb_2_updates = AlgorithmConfiguration(
    "L-BFGS-B",
    "L-BFGS-B with 2 Hessian updates",
    maxcor=2,
)
lbfgsb_20_updates = AlgorithmConfiguration(
    "L-BFGS-B",
    "L-BFGS-B with 20 Hessian updates",
    maxcor=20,
)

# Select the reference problems.
optimum = 0.0
target_values = TargetValues([10**-i for i in range(4, 7)] + [optimum])
doe_settings = {"doe_size": 5, "doe_algo_name": "OT_OPT_LHS"}
rastrigin_2d = Problem(
    "Rastrigin 2D",
    Rastrigin,
    target_values=target_values,
    optimum=optimum,
    **doe_settings,
    description="Rastrigin's 2-dimensional function.",
)
rosenbrock_2d = Problem(
    "Rosenbrock 2D",
    Rosenbrock,
    target_values=target_values,
    optimum=optimum,
    **doe_settings,
    description="Rosenbrock's 2-dimensional function.",
)
rosenbrock_5d = Problem(
    "Rosenbrock 5D",
    lambda: Rosenbrock(5),
    target_values=target_values,
    optimum=optimum,
    **doe_settings,
)

# Create the benchmarking scenario
output_directory = Path("scenario")
output_directory.mkdir(exist_ok=True)
Scenario(
    [
        AlgorithmsConfigurations(slsqp, name="SLSQP"),
        AlgorithmsConfigurations(lbfgsb_2_updates, lbfgsb_20_updates, name="L-BFGS-B"),
    ],
    output_directory,
).execute(
    [
        ProblemsGroup("Rastrigin", [rastrigin_2d], "Rastrigin's function."),
        ProblemsGroup(
            "Rosenbrock",
            [rosenbrock_2d, rosenbrock_5d],
            description="Rosenbrock's function for various dimensions.",
        ),
        ProblemsGroup(
            "Rastrigin & Rosenbrock",
            [rastrigin_2d, rosenbrock_2d, rosenbrock_5d],
        ),
    ],
)
