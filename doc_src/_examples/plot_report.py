"""
Generate a benchmarking report
==============================
"""
# %%
# In this example,
# we generate a **benchmarking report**
# based on the performances of three algorithms configurations
# on three reference problems.
#
# Imports
# -------
# We start by making the necessary imports.
from __future__ import annotations

import shutil
from pathlib import Path

from gemseo import configure
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

# %%
# Set the algorithms configurations
# ---------------------------------
# Let us define the algorithms configurations that we want to benchmark.
#
# For example,
# let us choose a configuration of the L-BFGS-B algorithm
# with a number of Hessian corrections limited to 2.
# (this option is called ``maxcor``.)
lbfgsb_2_corrections = AlgorithmConfiguration(
    "L-BFGS-B",
    "L-BFGS-B with 2 Hessian corrections",
    maxcor=2,
)
# %%
# Note:
#     The customized name "L-BFGS-B with 2 Hessian corrections"
#     will serve to refer to this algorithm configuration in the report.
#
# To investigate the influence of the `maxcor` option,
# let us consider a different configuration of L-BFGS-B
# with up to 20 Hessian corrections.
lbfgsb_20_corrections = AlgorithmConfiguration(
    "L-BFGS-B",
    "L-BFGS-B with 20 Hessian corrections",
    maxcor=20,
)
# %%
# Let us put these two configurations of L-BFGS-B
# in a same group of algorithms configurations
# so that a section of the report will be dedicated to them.
lbfgsb_configurations = AlgorithmsConfigurations(
    lbfgsb_2_corrections,
    lbfgsb_20_corrections,
    name="L-BFGS-B configurations",
)
# %%
# Additionally,
# let us choose the SLSQP algorithm,
# with all its options set to their default values,
# to compare it against L-BFGS-B.
# Let us put it in a group of its own.
slsqp_default = AlgorithmConfiguration("SLSQP")
slsqp_configurations = AlgorithmsConfigurations(slsqp_default, name="SLSQP")
# %%
# Set the reference problems
# --------------------------
# Let us choose two problems already implemented in GEMSEO as references
# to measure the performances of our selection of algorithms configurations:
# :class:`~gemseo.problems.analytical.rastrigin.Rastrigin`
# and :class:`~gemseo.problems.analytical.rosenbrock.Rosenbrock`.
#
# We define target values as an exponential scale of values decreasing towards zero,
# the minimum value of both Rastrigin's and Rosenbrock's functions.
optimum = 0.0
target_values = TargetValues([10**-i for i in range(4, 7)] + [optimum])
# %%
# N.B. it could be preferable to customize a different scale of target values
# for each problem,
# although we keep it simple here.
#
# We now have all the elements to define the benchmarking problems.
rastrigin_2d = Problem(
    "Rastrigin",
    Rastrigin,
    optimum=optimum,
    doe_size=5,
    doe_algo_name="OT_OPT_LHS",
    target_values=target_values,
)
rosenbrock_2d = Problem(
    "Rosenbrock",
    Rosenbrock,
    optimum=optimum,
    doe_size=5,
    doe_algo_name="OT_OPT_LHS",
    target_values=target_values,
)
# %%
# Here we configure a design of experiments (DOE)
# to generate five starting points by optimized Latin hypercube sampling (LHS).
#
# Let us gather these two two-variables problems in a group
# so that they will be treated together.
problems_2d = ProblemsGroup("2D problems", [rastrigin_2d, rosenbrock_2d])
# %%
# We add a five-variables problem, also based on Rosenbrock's function,
# to compare the algorithms configurations in higher dimension.
# Let us put it in a group of its own.
rosenbrock_5d = Problem(
    "Rosenbrock 5D",
    lambda: Rosenbrock(5),
    target_values=target_values,
    optimum=optimum,
    doe_size=5,
    doe_algo_name="OT_OPT_LHS",
)
problems_5d = ProblemsGroup("5D problems", [rosenbrock_5d])
# %%
# Generate the benchmarking results
# ---------------------------------
# Now that the algorithms configurations and the reference problems are properly set,
# we can measure the performances of the former on the latter.
#
# We set up a :class:`.Scenario` with
# our two groups of algorithms configurations
# and a path to a directory where to save the performance histories and the report.
scenario_dir = Path("performance_histories")
scenario_dir.mkdir()
scenario = Scenario([lbfgsb_configurations, slsqp_configurations], scenario_dir)
# %%
# Here we choose to deactivate the functions counters, progress bars and bounds check
# of GEMSEO to accelerate the script.
configure(
    activate_function_counters=False,
    activate_progress_bar=False,
    check_desvars_bounds=False,
)
# %%
# Let us execute the benchmarking scenario on our two groups of reference problems.
scenario.execute([problems_2d, problems_5d])
# %%
# The root the HTML report is the following.
str((scenario_dir / "report" / "_build" / "html" / "index.html").absolute())
# %%
# Here we remove the data as we do not intend to keep it.
shutil.rmtree(scenario_dir)
