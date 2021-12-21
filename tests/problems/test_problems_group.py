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
"""Tests for the problems group."""
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_benchmark.data_profiles.target_values import TargetValues
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from numpy import zeros
from pytest import raises


def test_is_algorithm_suited():
    """Check the assessment of the suitability of an algorithm to a problems group."""
    # Check a suited algorithm
    rosenbrock = Problem("Rosenbrock", Rosenbrock, [zeros(2)])
    group = ProblemsGroup("group", [rosenbrock])
    assert group.is_algorithm_suited("L-BFGS-B")
    # Check an ill-suited algorithm
    power2 = Problem("Power2", Power2, [zeros(3)])
    group = ProblemsGroup("another group", [rosenbrock, power2])
    assert not group.is_algorithm_suited("L-BFGS-B")


def test_compute_targets():
    """Check the computation of target values."""
    rosenbrock = Problem("Rosenbrock", Rosenbrock, [zeros(2)])
    with raises(ValueError, match="The benchmarking problem has no target value."):
        rosenbrock.target_values
    ProblemsGroup("group", [rosenbrock]).compute_targets(2, {"L-BFGS-B": {}})
    assert isinstance(rosenbrock.target_values, TargetValues)
