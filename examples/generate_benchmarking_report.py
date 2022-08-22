# -*- coding: utf-8 -*-
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

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A simple script to generate a benchmarking report."""
from pathlib import Path

from gemseo.problems.analytical.rastrigin import Rastrigin
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import \
    AlgorithmsConfigurations
from gemseo_benchmark.benchmarker import Benchmarker
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.report.report import Report

# Select the algorithms configurations to be benchmarked.
l_bfgs_b = AlgorithmConfiguration("L-BFGS-B")
slsqp = AlgorithmConfiguration("SLSQP")
algorithms_configurations = AlgorithmsConfigurations(l_bfgs_b, slsqp)

# Select the reference problems.
optimum = 0.0
target_values = TargetValues([10 ** -i for i in range(4, 7)] + [optimum])
rastrigin = Problem(
    "Rastrigin 2D", Rastrigin, target_values=target_values, optimum=optimum
)
rosenbrock = Problem(
    "Rosenbrock 2D", Rosenbrock, target_values=target_values, optimum=optimum
)
reference_problems = [rastrigin, rosenbrock]

# Run the algorithms on the reference problems.
histories_dir = Path(__file__).parent / "histories"
histories_dir.mkdir()
benchmarker = Benchmarker(histories_dir)
results = benchmarker.execute(reference_problems, algorithms_configurations)

# Generate the benchmarking report in HTML format.
report_dir = Path(__file__).parent.absolute() / "report"
group = ProblemsGroup("Unconstrained problems", reference_problems)
report = Report(report_dir, algorithms_configurations, [group], results)
report.generate()
