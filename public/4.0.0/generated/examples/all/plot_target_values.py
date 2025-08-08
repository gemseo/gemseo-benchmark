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
Generate target values
======================
"""

# %%
# In this example,
# we generate **target values** for an optimization problem configuration
# based on the performances of an algorithm configuration.
#
# Imports
# -------
# We start by making the necessary imports.
from __future__ import annotations

from gemseo import compute_doe
from gemseo import configure
from gemseo.problems.optimization.power_2 import Power2

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.problems.optimization_problem_configuration import (
    OptimizationProblemConfiguration,
)

# %%
# Let us consider the optimization problem
# [Power2][gemseo.problems.optimization.power_2.Power2]
# already implemented in GEMSEO.
problem = OptimizationProblemConfiguration(
    "Power2", Power2, optimum=Power2.get_solution()[1]
)
# %%
# We define ten starting points by optimized Latin hypercube sampling (LHS).
design_space = problem.create_problem().design_space
problem.starting_points = compute_doe(
    design_space, algo_name="OT_OPT_LHS", n_samples=10
)
# %%
# Let use the optimizer COBYLA to generate performance histories on the problem.
algorithms_configurations = AlgorithmsConfigurations(
    AlgorithmConfiguration(
        "NLOPT_COBYLA",
        max_iter=65,
        eq_tolerance=1e-4,
        ineq_tolerance=0.0,
        xtol_abs=0,
        xtol_rel=0,
        ftol_abs=0,
        ftol_rel=0,
    )
)
# %%
# Here we choose to deactivate the functions counters, progress bars and bounds check
# of GEMSEO to accelerate the script.
configure(
    enable_function_statistics=False,
    enable_progress_bar=False,
    check_desvars_bounds=False,
)
# %%
# Let us compute five target values for the problem.
# This automatic procedure has two stages:
#
# 1. execution of the specified algorithm configurations
#    once for each of the starting points,
# 2. automatic selection of target values based on the performance histories.
#
# These target values represent the milestones of the problem resolution.
problem.compute_target_values(5, algorithms_configurations, best_target_tolerance=1e-5)
# %%
# We can plot the performace histories used as reference
# for the computation of the target values,
# with the objective value on the vertical axis
# and the number of functions evaluations on the horizontal axis.
problem.targets_generator.plot_histories(problem.optimum, show=True)
# %%
# Finally, we can plot the target values:
# the objective value of each of the five targets is represented
# on the vertical axis with a marker indicating whether the target is feasible or not.
problem.target_values.plot()
