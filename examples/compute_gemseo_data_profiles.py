"""Compute data profiles for GEMSEO optimizers."""
from __future__ import annotations

from pathlib import Path
from shutil import rmtree

from gemseo.api import execute_algo
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.results.performance_history import PerformanceHistory
from gemseo_benchmark.results.results import Results

# Set the reference problem
problem = Problem("Rosenbrock", Rosenbrock, doe_algo_name="DiagonalDOE", doe_size=10)

# Set the algorithms to be compared
algorithms_configurations = AlgorithmsConfigurations(
    AlgorithmConfiguration("SLSQP"),
    AlgorithmConfiguration("DUAL_ANNEALING"),
    AlgorithmConfiguration("L-BFGS-B"),
)

# Run the algorithms on the reference problem
histories_dir = Path(__file__).parent / "gemseo_histories"
histories_dir.mkdir()
results = Results()
for algorithm_configuration in algorithms_configurations:
    algo_name = algorithm_configuration.algorithm_name
    for index, opt_problem in enumerate(problem):
        execute_algo(
            opt_problem, algo_name, **algorithm_configuration.algorithm_options
        )
        objective_values, measures, feasibility_statuses = problem.compute_performance(
            opt_problem
        )
        history = PerformanceHistory(objective_values, measures, feasibility_statuses)
        history_path = histories_dir / f"{algo_name}_{index + 1}.json"
        history.to_file(history_path)
        results.add_path(algo_name, problem.name, history_path)

# Generate target values and generate the data profile
problem_group = ProblemsGroup("Reference group", [problem])
reference_algorithm_configurations = AlgorithmsConfigurations(
    AlgorithmConfiguration("DIFFERENTIAL_EVOLUTION")
)
problem_group.compute_targets(3, algorithms_configurations)
problem_group.compute_data_profile(algorithms_configurations, results, show=True)

# Clean up the histories files
rmtree(histories_dir)
