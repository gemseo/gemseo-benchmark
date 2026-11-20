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
from gemseo.doe import CustomDOE_Settings
from gemseo.optimization import L_BFGS_B_Settings
from gemseo.problem.optimization.rosenbrock import Rosenbrock
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
from tests.problems.conftest import test_compute_data_profile_files_parametrize

algorithms_configurations = AlgorithmsConfigurations(
    AlgorithmConfiguration(L_BFGS_B_Settings())
)


def test_compute_target_values():
    """Check the computation of target values."""
    rosenbrock = OptimizationProblemConfiguration(
        "Rosenbrock", Rosenbrock, doe_settings=CustomDOE_Settings(samples=zeros((1, 2)))
    )
    with pytest.raises(
        ValueError, match=re.escape("The problem configuration has no target value.")
    ):
        _ = rosenbrock.target_values
    ProblemsGroup("group", [rosenbrock]).compute_target_values(
        2, algorithms_configurations
    )
    assert isinstance(rosenbrock.target_values, TargetValues)


@pytest.mark.parametrize(
    "use_abscissa_log_scale", [False, True], ids=["linear_scale", "log_scale"]
)
def test_compute_data_profile(
    problem_a, problem_b, results, use_abscissa_log_scale, snapshot_matplotlib
):
    """Check the computation of data profiles."""
    group = ProblemsGroup("A group", [problem_a, problem_b])
    figures = group.compute_data_profile(
        algorithms_configurations,
        results,
        show=False,
        max_eval_number=5,
        use_abscissa_log_scale=use_abscissa_log_scale,
        use_time_log_scale=use_abscissa_log_scale,
    )
    for figure in figures:
        assert figure == snapshot_matplotlib


@test_compute_data_profile_files_parametrize
def test_compute_data_profile_files(
    tmp_path, problem_a, problem_b, results, use_subplots, number_of_figures, file_names
):
    """Check the files of the data profiles of a group of problems."""
    path = tmp_path / "data_profile.png"
    figures = ProblemsGroup("A group", [problem_a, problem_b]).compute_data_profile(
        algorithms_configurations,
        results,
        show=False,
        plot_path=path,
        use_subplots=use_subplots,
    )
    assert len(figures) == number_of_figures
    assert {path.name for path in tmp_path.iterdir()} == file_names


def test_iter(problem_a, problem_b):
    """Check the iteration over a group of problems."""
    group = ProblemsGroup("A group", [problem_a, problem_b])
    assert list(group) == [problem_a, problem_b]
