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
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.settings.base_settings import BaseSettings
from pydantic import Field

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration


class Dummy_Settings(BaseSettings):  # noqa: N801
    """Minimal settings for a fake algorithm used in tests."""

    _TARGET_CLASS_NAME = "Dummy"

    max_iter: int = Field(default=100)


@pytest.mark.parametrize(
    ("input_name", "output_name"),
    [("SciPy SLSQP", "SciPy SLSQP"), (None, "SLSQP_max_iter=9")],
)
def test_name(input_name, output_name):
    """Check the name of an algorithm configuration."""
    algorithm_configuration = AlgorithmConfiguration(
        SLSQP_Settings(max_iter=9), input_name
    )
    assert algorithm_configuration.name == output_name


@pytest.mark.parametrize(
    ("input_name", "output_name"),
    [("SciPy SLSQP", "SciPy SLSQP"), (None, "SLSQP_max_iter=9")],
)
@pytest.mark.parametrize("skip_instance_algorithm_options", [False, True])
def test_to_dict(input_name, output_name, skip_instance_algorithm_options):
    """Check the export of an algorithm configuration as a dictionary."""
    algorithm_configuration = AlgorithmConfiguration(
        SLSQP_Settings(max_iter=9), input_name, {"seed": lambda index: index}
    )
    dictionary = algorithm_configuration.to_dict(skip_instance_algorithm_options)
    assert dictionary["algorithm_settings_class"] == (
        "gemseo.algos.opt.scipy_local.settings.slsqp.SLSQP_Settings"
    )
    assert dictionary["configuration_name"] == output_name
    assert dictionary["algorithm_settings"] == {"max_iter": 9}
    if skip_instance_algorithm_options:
        assert dictionary.keys() == {
            "algorithm_settings_class",
            "configuration_name",
            "algorithm_settings",
        }
    else:
        assert dictionary.keys() == {
            "algorithm_settings_class",
            "configuration_name",
            "algorithm_settings",
            "instance_algorithm_options",
        }
        assert dictionary["instance_algorithm_options"].keys() == {"seed"}
        assert dictionary["instance_algorithm_options"]["seed"](7) == 7


def test_from_dict():
    """Check the import of an algorithm configuration from a dictionary."""
    algorithm_configuration = AlgorithmConfiguration.from_dict({
        "configuration_name": "SciPy SLSQP",
        "algorithm_settings_class": (
            "gemseo.algos.opt.scipy_local.settings.slsqp.SLSQP_Settings"
        ),
        "algorithm_settings": {"max_iter": 9},
        "instance_algorithm_options": {"seed": lambda index: index},
    })
    assert algorithm_configuration.name == "SciPy SLSQP"
    assert algorithm_configuration.algorithm_name == "SLSQP"
    assert algorithm_configuration.algorithm_options == {"max_iter": 9}


def test_copy(algorithm_configuration):
    """Check the copy an algorithm configuration."""
    algorithm_configuration = AlgorithmConfiguration(
        Dummy_Settings(max_iter=9), "Algorithm configuration"
    ).copy()
    assert algorithm_configuration.algorithm_name == "Dummy"
    assert algorithm_configuration.name == "Algorithm configuration"
    assert algorithm_configuration.algorithm_options == {"max_iter": 9}
