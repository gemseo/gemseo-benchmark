"""Tests for the algorithm configuration."""
import pytest

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration


@pytest.mark.parametrize(
    ["input_name", "output_name"],
    [("SciPy SLSQP", "SciPy SLSQP"), (None, "SLSQP_max_iter=9")]
)
def test_name(input_name, output_name):
    """Check the name of an algorithm configuration."""
    algorithm_configuration = AlgorithmConfiguration("SLSQP", input_name, max_iter=9)
    assert algorithm_configuration.name == output_name


@pytest.mark.parametrize(
    ["input_name", "output_name"],
    [("SciPy SLSQP", "SciPy SLSQP"), (None, "SLSQP_max_iter=9")]
)
def test_to_dict(input_name, output_name):
    """Check the export of an algorithm configuration as a dictionary."""
    algorithm_configuration = AlgorithmConfiguration("SLSQP", input_name, max_iter=9)
    assert algorithm_configuration.to_dict() == {
        "configuration_name": output_name, "algorithm_name": "SLSQP",
        "algorithm_options": {"max_iter": 9}
    }


def test_from_dict():
    """Check the import of an algorithm configuration from a dictionary."""
    algorithm_configuration = AlgorithmConfiguration.from_dict(
        {
            "configuration_name": "SciPy SLSQP", "algorithm_name": "SLSQP",
            "algorithm_options": {"max_iter": 9}
        }
    )
    assert algorithm_configuration.name == "SciPy SLSQP"
    assert algorithm_configuration.algorithm_name == "SLSQP"
    assert algorithm_configuration.algorithm_options == {"max_iter": 9}
