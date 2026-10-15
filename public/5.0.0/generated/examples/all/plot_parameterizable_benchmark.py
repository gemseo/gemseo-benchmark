# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""
# Use a parametric benchmark problem
"""

# %%
# In this example,
# we use the scalable Rosenbrock problem
# to benchmark two optimization algorithms for increasing dimensions.
#
# ## Imports
# We start by making the necessary imports.
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from gemseo import configuration
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.algos.opt.scipy_local.settings.lbfgsb import L_BFGS_B_Settings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problems.optimization.rosenbrock import Rosenbrock

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems.optimization_problem_configuration import (
    OptimizationProblemConfiguration,
)
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.scenario import Scenario

# %%
# We use the fast logging configuration of GEMSEO.
configuration.fast = True


# %%
# ## Create a generator of optimization problems
#
# The second argument of `OptimizationProblemConfiguration` must be
# a callable that returns an optimization problem and takes *no arguments*.
# In the case of the scalable Rosenbrock problem,
# we can use a functor depending on the input dimension:
class RosenbrockCreator:
    """Create a Rosenbrock problem."""

    def __init__(self, n_x: int) -> None:
        """
        Args:
            n_x: The input dimension.
        """
        self.n_x = n_x

    def __call__(self) -> Rosenbrock:
        # Note that this method does not accept arguments.
        return Rosenbrock(n_x=self.n_x)


# %%
# ## Set the reference problems
#
# We define target values as an exponential scale of values decreasing towards zero,
# the minimum value of the Rosenbrock function.
def generate_problems(n_x: int) -> ProblemsGroup:
    """Generate a group of problems.

    Args:
        n_x: The input dimension.
    """
    optimum = 0.0
    target_values = TargetValues([10**-i for i in range(4, 7)] + [optimum])
    problem_creator = RosenbrockCreator(n_x)
    problem_configuration = OptimizationProblemConfiguration(
        f"Rosenbrock_{dimension}D",
        problem_creator,
        optimum=optimum,
        doe_settings=OT_OPT_LHS_Settings(n_samples=5),
        target_values=target_values,
    )
    return ProblemsGroup(f"{n_x}D_problems", [problem_configuration])


# %%
# ## Set the algorithms configurations
#
# Finally, we gather our selection of algorithms configurations in a group.
#
# Let us define the algorithms configurations
# for which we want to compute data profiles.
#
# For example,
# let us choose the L-BFGS-B and SLSQP algorithms.
algorithms_configurations = AlgorithmsConfigurations(
    AlgorithmConfiguration(L_BFGS_B_Settings()),
    AlgorithmConfiguration(SLSQP_Settings()),
    name="Derivative-based algorithms",
)
# %%
# ## Generate the benchmarking results
#
# Now that the algorithms configurations and the reference problems are properly set,
# we can measure the performances of the former on the latter.
#
# We set up a [Scenario][gemseo_benchmark.scenario.Scenario] with
# our group of algorithms configurations
# and a path to a directory where to save the performance histories.
scenario_dir = Path(tempfile.mkdtemp())
scenario = Scenario([algorithms_configurations], scenario_dir)
# %%
# ## Execute and compute the datas profiles
#
# Let us execute the benchmarking scenario for various values of the parameter.
#
# Note:
#     Here we skip the generation of the report
#     as we only intend to compute the data profiles.
for dimension in [2, 5, 10, 20]:
    problems = generate_problems(dimension)
    results = scenario.execute([problems], skip_report=True)
    problems.compute_data_profile(algorithms_configurations, results, show=True)
# %%
# Now that the performances histories are generated for the reference problems,
# the data profiles of the algorithms configurations can be computed.
#
# Finally we remove the performances histories as we do not wish to keep them.
shutil.rmtree(scenario_dir)
