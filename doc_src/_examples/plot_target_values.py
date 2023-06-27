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
# we generate **target values** for a problem
# based on the performances of an algorithm configuration.
#
# Imports
# -------
# We start by making the necessary imports.
from __future__ import annotations

from gemseo import compute_doe
from gemseo import configure
from gemseo.problems.analytical.power_2 import Power2
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.problems.problem import Problem

# %%
# Let us consider the problem :class:`~gemseo.problems.analytical.power_2.Power2`
# already implemented in GEMSEO.
problem = Problem(
    name="Power2",
    optimization_problem_creator=Power2,
    optimum=Power2.get_solution()[1],
)
# %%
# We define ten starting points by optimized Latin hypercube sampling (LHS).
design_space = problem.creator().design_space
problem.start_points = compute_doe(design_space, "OT_OPT_LHS", 10)
# %%
# Let use the optimizer COBYLA to generate performance histories on the problem.
algorithms_configurations = AlgorithmsConfigurations(
    AlgorithmConfiguration(
        "NLOPT_COBYLA",
        max_iter=65,
        eq_tolerance=1e-4,
        ineq_tolerance=0.0,
    )
)
# %%
# Here we choose to deactivate the functions counters, progress bars and bounds check
# of GEMSEO to accelerate the script.
configure(
    activate_function_counters=False,
    activate_progress_bar=False,
    check_desvars_bounds=False,
)
# %%
# Let us compute five target values for the problem.
# This automatic procedure has two stages:
#
# #. execution of the specified algorithms once for each of the starting points,
# #. automatic selection of target values based on the algorithms histories.
#
# These targets represent the milestones of the problem resolution.
problem.compute_targets(5, algorithms_configurations, best_target_tolerance=1e-5)
# %%
# We can plot the algorithms histories used as reference
# for the computation of the target values,
# with the objective value on the vertical axis
# and the number of functions evaluations on the horizontal axis.
problem.targets_generator.plot_histories(problem.optimum, show=True)
# %%
# Finally, we can plot the target values:
# the objective value of each of the five targets is represented
# on the vertical axis with a marker indicating whether the target is feasible or not.
problem.target_values.plot()
