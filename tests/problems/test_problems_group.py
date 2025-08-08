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
"""Tests for the problems group."""

from __future__ import annotations

import re

import pytest
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from matplotlib import pyplot
from matplotlib.testing.decorators import image_comparison
from numpy import zeros

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems.optimization_problem_configuration import (
    OptimizationProblemConfiguration,
)
from gemseo_benchmark.problems.problems_group import ProblemsGroup

algorithms_configurations = AlgorithmsConfigurations(AlgorithmConfiguration("L-BFGS-B"))


def test_compute_target_values():
    """Check the computation of target values."""
    rosenbrock = OptimizationProblemConfiguration("Rosenbrock", Rosenbrock, [zeros(2)])
    with pytest.raises(
        ValueError, match=re.escape("The problem configuration has no target value.")
    ):
        _ = rosenbrock.target_values
    ProblemsGroup("group", [rosenbrock]).compute_target_values(
        2, algorithms_configurations
    )
    assert isinstance(rosenbrock.target_values, TargetValues)


@pytest.mark.parametrize(
    ("baseline_images", "use_abscissa_log_scale"),
    [
        (
            [f"data_profile[use_abscissa_log_scale={use_abscissa_log_scale}]"],
            use_abscissa_log_scale,
        )
        for use_abscissa_log_scale in [False, True]
    ],
)
@image_comparison(None, ["png"])
def test_compute_data_profile(
    baseline_images, problem_a, problem_b, results, use_abscissa_log_scale
):
    """Check the computation of data profiles."""
    group = ProblemsGroup("A group", [problem_a, problem_b])
    pyplot.close("all")
    group.compute_data_profile(
        algorithms_configurations,
        results,
        show=False,
        max_eval_number=5,
        use_abscissa_log_scale=use_abscissa_log_scale,
    )


def test_iter(problem_a, problem_b):
    """Check the iteration over a group of problems."""
    group = ProblemsGroup("A group", [problem_a, problem_b])
    assert list(group) == [problem_a, problem_b]
