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
"""Tests for the algorithm configuration."""
from __future__ import annotations

import pytest
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration


@pytest.mark.parametrize(
    ["input_name", "output_name"],
    [("SciPy SLSQP", "SciPy SLSQP"), (None, "SLSQP_max_iter=9")],
)
def test_name(input_name, output_name):
    """Check the name of an algorithm configuration."""
    algorithm_configuration = AlgorithmConfiguration("SLSQP", input_name, max_iter=9)
    assert algorithm_configuration.name == output_name


@pytest.mark.parametrize(
    ["input_name", "output_name"],
    [("SciPy SLSQP", "SciPy SLSQP"), (None, "SLSQP_max_iter=9")],
)
def test_to_dict(input_name, output_name):
    """Check the export of an algorithm configuration as a dictionary."""
    algorithm_configuration = AlgorithmConfiguration("SLSQP", input_name, max_iter=9)
    assert algorithm_configuration.to_dict() == {
        "configuration_name": output_name,
        "algorithm_name": "SLSQP",
        "algorithm_options": {"max_iter": 9},
    }


def test_from_dict():
    """Check the import of an algorithm configuration from a dictionary."""
    algorithm_configuration = AlgorithmConfiguration.from_dict(
        {
            "configuration_name": "SciPy SLSQP",
            "algorithm_name": "SLSQP",
            "algorithm_options": {"max_iter": 9},
        }
    )
    assert algorithm_configuration.name == "SciPy SLSQP"
    assert algorithm_configuration.algorithm_name == "SLSQP"
    assert algorithm_configuration.algorithm_options == {"max_iter": 9}
