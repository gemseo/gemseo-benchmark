from gemseo_benchmark.data_profiles import PerformanceHistory
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.utils.py23_compat import Path
from gemseo_benchmark.problems import Problem
from gemseo_benchmark.problems import ProblemsGroup

# Set the reference problem
problem = Problem("Rosenbrock", Rosenbrock, doe_algo_name="DiagonalDOE", doe_size=10)

# Set the algorithms to be compared
comparison_algos_specs = {
    "SLSQP": dict(),
    "DUAL_ANNEALING": dict(),
    "L-BFGS-B": dict()
}

# Run the algorithms on the reference problem
histories_paths = {
    algo_name: {"Rosenbrock": list()} for algo_name in comparison_algos_specs
}
for algo_name, algo_options in comparison_algos_specs.items():
    for index, instance in enumerate(problem):
        OptimizersFactory().execute(instance, algo_name, **algo_options)
        obj_values, measures, feas_statuses = problem.compute_performance(instance)
        history = PerformanceHistory(obj_values, measures, feas_statuses)
        history_path = Path(__file__).parent / "{}_{}.json".format(algo_name, index + 1)
        history.to_file(history_path)
        histories_paths[algo_name][problem.name].append(history_path)

# Generate target values and generate the data profile
group = ProblemsGroup("Reference group", [problem])
ref_algos_specs = {"DIFFERENTIAL_EVOLUTION": dict()}
group.compute_targets(3, ref_algos_specs)
group.compute_data_profile(comparison_algos_specs, histories_paths, show=True)

# Clean up the histories files
for algo_histories in histories_paths.values():
    for problem_histories in algo_histories.values():
        for history_path in problem_histories:
            history_path.unlink()
