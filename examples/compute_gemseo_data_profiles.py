from gemseo.problems.analytical.power_2 import Power2

from problems.problem import Problem

# Set the benchmarking reference problem
problem = Problem("Power2", Power2, doe_algo_name="lhs", doe_size=10)

# Generate target values
ref_algos_specs = {"NLOPT_COBYLA": dict()}
problem.generate_targets(3, ref_algos_specs, show=True)

# Generate data profiles
comparison_algos_specs = {
    "SLSQP": dict(),
    "NLOPT_SLSQP": dict(),
    "NLOPT_COBYLA": dict()
}
problem.generate_data_profile(comparison_algos_specs, show=True)
