from gemseo.api import get_available_opt_algorithms
from gemseo.problems.analytical.power_2 import Power2

from problems.bench_problem import BenchProblem
from problems.problems_group import ProblemsGroup

# Set the benchmarking reference problems
reference_problems = [
    BenchProblem("Power2", Power2, [Power2().design_space.get_current_x()])
]
problems_group = ProblemsGroup(reference_problems)

# Generate target values for each of the reference problems
targets_number = 20
reference_algorithms = {"SLSQP": dict()}
problems_group.generate_targets(targets_number, reference_algorithms, feasible=True)

# Generate data profiles
algo_names = get_available_opt_algorithms()
algorithms = dict.fromkeys(algo_names, dict())
for an_algo in algo_names:
    if not problems_group.is_algorithm_suited(an_algo):
        algorithms.pop(an_algo)
problems_group.generate_data_profile(algorithms)
