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

# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
import tempfile
from pathlib import Path

from gemseo import configure
from gemseo.problems.optimization.rastrigin import Rastrigin
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
# [Rastrigin][gemseo.problems.optimization.rastrigin.Rastrigin]
# and [Rosenbrock][gemseo.problems.optimization.rosenbrock.Rosenbrock].
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
# We now have all the elements to define the problem configurations.
rastrigin_2d = OptimizationProblemConfiguration(
    "Rastrigin",
    Rastrigin,
    optimum=optimum,
    doe_size=5,
    doe_algo_name="OT_OPT_LHS",
    target_values=target_values,
)
rosenbrock_2d = OptimizationProblemConfiguration(
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
def create_problem():
    return Rosenbrock(5)


rosenbrock_5d = OptimizationProblemConfiguration(
    "Rosenbrock 5D",
    create_problem,
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
# We set up a [Scenario][gemseo_benchmark.scenario.Scenario] with
# our two groups of algorithms configurations
# and a path to a directory where to save the performance histories and the report.
scenario_dir = Path(tempfile.mkdtemp())
scenario = Scenario([lbfgsb_configurations, slsqp_configurations], scenario_dir)
# %%
# Here we choose to deactivate the functions counters, progress bars and bounds check
# of GEMSEO to accelerate the script.
configure(
    enable_function_statistics=False,
    enable_progress_bar=False,
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
