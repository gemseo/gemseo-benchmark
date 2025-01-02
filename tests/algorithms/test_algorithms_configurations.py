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
"""Tests for the algorithms configurations."""

from __future__ import annotations

import re
from unittest import mock

import pytest

from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)


def test_init(algorithm_configuration):
    """Check the initialization of algorithms configurations."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The collection already contains an algorithm configuration named "
            f"{algorithm_configuration.algorithm_name}."
        ),
    ):
        AlgorithmsConfigurations(algorithm_configuration, algorithm_configuration)


@pytest.fixture(scope="module")
def configuration_a():
    """Algorithm configuration A."""
    config = mock.Mock()
    config.name = "Configuration A"
    config.algorithm_name = "Algorithm A"
    return config


@pytest.fixture(scope="module")
def configuration_b():
    """Algorithm configuration B."""
    config = mock.Mock()
    config.name = "Configuration B"
    config.algorithm_name = "Algorithm B"
    return config


@pytest.fixture(scope="module")
def configuration_c():
    """Algorithm configuration C."""
    config = mock.Mock()
    config.name = "Configuration C"
    config.algorithm_name = "Algorithm C"
    return config


def test_names(configuration_b, configuration_c, configuration_a):
    """Check the access to the names of the algorithms configurations."""
    algorithms_configurations = AlgorithmsConfigurations(
        configuration_b, configuration_c, configuration_a
    )
    assert algorithms_configurations.names == [
        configuration_a.name,
        configuration_b.name,
        configuration_c.name,
    ]


def test_algorithms(configuration_b, configuration_c, configuration_a):
    """Check the access to the names of the algorithms."""
    algorithms_configurations = AlgorithmsConfigurations(
        configuration_b, configuration_c, configuration_a
    )
    assert algorithms_configurations.algorithms == [
        configuration_a.algorithm_name,
        configuration_b.algorithm_name,
        configuration_c.algorithm_name,
    ]


def test_configurations(configuration_b, configuration_c, configuration_a):
    """Check the access to the algorithms configurations."""
    algorithms_configurations = AlgorithmsConfigurations(
        configuration_b, configuration_c, configuration_a
    )
    assert algorithms_configurations.configurations == [
        configuration_a,
        configuration_b,
        configuration_c,
    ]


def test_discard(configuration_b, configuration_c, configuration_a):
    """Check the discarding of an algorithm configuration."""
    algorithms_configurations = AlgorithmsConfigurations(
        configuration_b, configuration_c, configuration_a
    )
    algorithms_configurations.discard(configuration_a)
    assert algorithms_configurations.configurations == [
        configuration_b,
        configuration_c,
    ]


def test_name(configuration_b, configuration_c, configuration_a):
    """Check the accessor to the name of a collection of algorithms configurations."""
    name = "name of the collection"
    algorithms_configurations = AlgorithmsConfigurations(
        configuration_b,
        configuration_c,
        configuration_a,
        name=name,
    )
    assert algorithms_configurations.name == name


def test_unnamed_collection(configuration_b, configuration_c, configuration_a):
    """Check the accessor to the name of an unnamed collection of configurations."""
    algorithms_configurations = AlgorithmsConfigurations(
        configuration_b,
        configuration_c,
        configuration_a,
    )
    with pytest.raises(
        ValueError,
        match=re.escape("The collection of algorithms configurations has no name."),
    ):
        algorithms_configurations.name  # noqa: B018
