"""Tests for the algorithms configurations."""
from unittest import mock

import pytest

from gemseo_benchmark.algorithms.algorithms_configurations import \
    AlgorithmsConfigurations


def test_init(algorithm_configuration):
    """Check the initialization of algorithms configurations."""
    with pytest.raises(
            ValueError,
            match="The collection already contains an algorithm configuration named "
                  f"{algorithm_configuration.algorithm_name}."
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
        configuration_a.name, configuration_b.name, configuration_c.name
    ]


def test_algorithms(configuration_b, configuration_c, configuration_a):
    """Check the access to the names of the algorithms."""
    algorithms_configurations = AlgorithmsConfigurations(
        configuration_b, configuration_c, configuration_a
    )
    assert algorithms_configurations.algorithms == [
        configuration_a.algorithm_name, configuration_b.algorithm_name,
        configuration_c.algorithm_name
    ]


def test_configurations(configuration_b, configuration_c, configuration_a):
    """Check the access to the algorithms configurations."""
    algorithms_configurations = AlgorithmsConfigurations(
        configuration_b, configuration_c, configuration_a
    )
    assert algorithms_configurations.configurations == [
        configuration_a, configuration_b, configuration_c
    ]


def test_discard(configuration_b, configuration_c, configuration_a):
    """Check the discarding of an algorithm configuration."""
    algorithms_configurations = AlgorithmsConfigurations(
        configuration_b, configuration_c, configuration_a
    )
    algorithms_configurations.discard(configuration_a)
    assert algorithms_configurations.configurations == [
        configuration_b, configuration_c
    ]
