from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.utils.py23_compat import Path

from data_profiles.performance_history import PerformanceHistory
from problems.problem import Problem
from problems.problems_group import ProblemsGroup

# Set the reference problem
problem = Problem("Power2", Power2, doe_algo_name="lhs", doe_size=10)

# Set the algorithms to be compared
comparison_algos_specs = {
    "SLSQP": dict(),
    "NLOPT_SLSQP": dict(),
    "NLOPT_COBYLA": dict()
}

# Run the algorithms on the reference problem
histories_paths = {
    algo_name: {"Power2": list()} for algo_name in comparison_algos_specs
}
for algo_name, algo_options in comparison_algos_specs.items():
    for index, instance in enumerate(problem):
        OptimizersFactory().execute(problem, algo_name, **algo_options)
        obj_values, measures, feas_statuses = problem.compute_performance(instance)
        history = PerformanceHistory(obj_values, measures, feas_statuses)
        history_path = Path(__file__).parent / "history_{}.json".format(index + 1)
        history.to_file(history_path)
        histories_paths[algo_name][problem.name].append(history_path)

# Generate target values and generate the data profile
group = ProblemsGroup("Reference group", [problem])
ref_algos_specs = {"NLOPT_COBYLA": dict()}
group.compute_targets(3, ref_algos_specs)
group.compute_data_profile(comparison_algos_specs, histories_paths, show=True)
