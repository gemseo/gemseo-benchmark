"""Compute data profiles for GEMSEO optimizers."""
from shutil import rmtree

from gemseo.api import execute_algo
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.utils.py23_compat import Path
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.results.performance_history import PerformanceHistory
from gemseo_benchmark.results.results import Results

# Set the reference problem
problem = Problem("Rosenbrock", Rosenbrock, doe_algo_name="DiagonalDOE", doe_size=10)

# Set the algorithms to be compared
comparison_algos_specs = {
    "SLSQP": dict(),
    "DUAL_ANNEALING": dict(),
    "L-BFGS-B": dict()
}

# Run the algorithms on the reference problem
histories_dir = Path(__file__).parent / "gemseo_histories"
histories_dir.mkdir()
results = Results()
for algo_name, algo_options in comparison_algos_specs.items():
    for index, instance in enumerate(problem):
        execute_algo(instance, algo_name, **algo_options)
        obj_values, measures, feas_statuses = problem.compute_performance(instance)
        history = PerformanceHistory(obj_values, measures, feas_statuses)
        history_path = histories_dir / f"{algo_name}_{index+1}.json"
        history.to_file(history_path)
        results.add_path(algo_name, problem.name, history_path)

# Generate target values and generate the data profile
group = ProblemsGroup("Reference group", [problem])
ref_algos_specs = {"DIFFERENTIAL_EVOLUTION": dict()}
group.compute_targets(3, ref_algos_specs)
group.compute_data_profile(comparison_algos_specs, results, show=True)

# Clean up the histories files
rmtree(histories_dir)
