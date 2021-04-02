from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.problems.analytical.power_2 import Power2

from data_profiles.data_profile import DataProfile
from data_profiles.targets_generator import TargetsGenerator


def extract_histories(
        problem  # type: OptimizationProblem
):  # type: (...) -> tuple
    """Extract the data profiles input from a Gemseo optimization problem.

    Args:
        problem: The Gemseo optimization problem.

    Returns:
        (
            The history of objective values,
            The history of feasibility,
            The history of constraints violations
        )

    """
    obj_name = problem.objective.name
    history = [(the_values[obj_name],) + problem.get_violation_criteria(an_x)
               for an_x, the_values in problem.database.items()]
    obj_history, feas_history, viol_history = zip(*history)
    return obj_history, feas_history, viol_history


# Set the benchmarking reference problems
ref_problems = {"Power2_default": Power2}

ref_algos = {"SLSQP": dict()}

# Generate target values for each of the reference problems
targets_number = 20
targets_values = dict()
for a_pb_name, a_pb_class in ref_problems.items():
    target_generator = TargetsGenerator()
    # Generate reference histories for the reference problem
    for an_algo_name, an_algo_options in ref_algos.items():
        pb_instance = a_pb_class()
        # FIXME: instantiate a new problem every time?
        OptimizersFactory().execute(pb_instance, an_algo_name, **an_algo_options)
        obj_hist, feas_hist, meas_hist = extract_histories(pb_instance)
        target_generator.add_history(obj_hist, meas_hist, feas_hist)
    targets_values[a_pb_name] = target_generator.run(targets_number, plot=True)
# TODO: target values generator based on Gemseo

# Generate data profiles
data_profile = DataProfile(targets_values)
algorithms = {"NLOPT_COBYLA": dict()}
for an_algo_name, an_algo_options in algorithms.items():
    values_histories = dict()
    for a_pb_name, a_pb_class in ref_problems.items():
        pb_instance = a_pb_class()
        OptimizersFactory().execute(pb_instance, an_algo_name, **an_algo_options)
        obj_hist, feas_hist, meas_hist = extract_histories(pb_instance)
        data_profile.add_history(a_pb_name, an_algo_name, obj_hist, meas_hist,
                                 feas_hist)
data_profile.plot()
