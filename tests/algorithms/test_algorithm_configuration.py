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

import re
from unittest import mock

import pytest
from gemseo.optimization import SLSQP_Settings
from gemseo.util.pydantic import BaseSettings
from pydantic import Field
from pydantic import ValidationError

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration


class Dummy_Settings(BaseSettings):  # ruff: ignore[invalid-class-name]
    """Minimal settings for a fake algorithm used in tests."""

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
        SLSQP_Settings(max_iter=9),
        input_name,
        {"seed": lambda problem_configuration, index: index},
    )
    dictionary = algorithm_configuration.to_dict(skip_instance_algorithm_options)
    assert dictionary["algorithm_settings_class"] == (
        "gemseo.optimization.scipy_local.settings.slsqp.SLSQP_Settings"
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
        assert dictionary["instance_algorithm_options"]["seed"](mock.Mock(), 7) == 7


def test_from_dict():
    """Check the import of an algorithm configuration from a dictionary."""
    algorithm_configuration = AlgorithmConfiguration.from_dict({
        "configuration_name": "SciPy SLSQP",
        "algorithm_settings_class": (
            "gemseo.optimization.scipy_local.settings.slsqp.SLSQP_Settings"
        ),
        "algorithm_settings": {"max_iter": 9},
        "instance_algorithm_options": {
            "seed": lambda problem_configuration, index: index
        },
    })
    assert algorithm_configuration.name == "SciPy SLSQP"
    assert algorithm_configuration.algorithm_name == "SLSQP"
    assert algorithm_configuration.algorithm_options == {"max_iter": 9}


def test_from_dict_earlier_version():
    """Check the import of an algorithm configuration saved by gemseo-benchmark 4."""
    with pytest.raises(
        ValueError,
        match=(
            r"^The algorithm configuration misses the keys "
            r"'algorithm_settings_class' and 'algorithm_settings'; "
            r"it was probably written by an earlier version "
            r"of gemseo-benchmark and must be computed again\.$"
        ),
    ):
        AlgorithmConfiguration.from_dict({
            "configuration_name": "SciPy SLSQP",
            "algorithm_name": "SLSQP",
            "algorithm_options": {"max_iter": 9},
        })


@pytest.mark.parametrize(
    ("dictionary", "keys"),
    [
        (
            {"configuration_name": "SLSQP", "algorithm_settings": {}},
            "key 'algorithm_settings_class'",
        ),
        (
            {},
            (
                "keys 'configuration_name', 'algorithm_settings_class' "
                "and 'algorithm_settings'"
            ),
        ),
    ],
)
def test_from_dict_missing_keys(dictionary, keys):
    """Check the error message when an algorithm configuration misses keys."""
    with pytest.raises(
        ValueError,
        match=rf"^The algorithm configuration misses the {keys}; ",
    ):
        AlgorithmConfiguration.from_dict(dictionary)


@pytest.mark.parametrize(
    "settings_class",
    [
        "gemseo.algos.opt.mma.settings.mma_settings.MMASVanberg_Settings",
        "gemseo.optimization.scipy_local.settings.slsqp.Unknown_Settings",
    ],
)
def test_from_dict_unknown_settings_class(settings_class):
    """Check the import of an algorithm configuration with an unknown settings class."""
    with pytest.raises(
        ValueError,
        match=(
            rf"^The class of the algorithm settings '{re.escape(settings_class)}' "
            r"cannot be imported; .* must be computed again\.$"
        ),
    ) as error:
        AlgorithmConfiguration.from_dict({
            "configuration_name": "MMA",
            "algorithm_settings_class": settings_class,
            "algorithm_settings": {},
        })

    assert isinstance(error.value.__cause__, (ImportError, AttributeError))


def test_get_instance_settings():
    """Check the settings of the algorithm for an instance of a problem."""
    settings = SLSQP_Settings(max_iter=9)
    algorithm_configuration = AlgorithmConfiguration(
        settings,
        instance_algorithm_options={
            "max_iter": lambda problem_configuration, index: (
                problem_configuration.dimension + index
            )
        },
    )
    instance_settings = algorithm_configuration.get_instance_settings(
        mock.Mock(dimension=10), 2
    )
    assert isinstance(instance_settings, SLSQP_Settings)
    assert instance_settings.max_iter == 12
    assert instance_settings.model_fields_set == {"max_iter"}
    assert settings.max_iter == 9


def test_get_instance_settings_without_instance_options():
    """Check the settings of the algorithm without options specific to instances."""
    settings = SLSQP_Settings(max_iter=9)
    instance_settings = AlgorithmConfiguration(settings).get_instance_settings(
        mock.Mock(), 0
    )
    assert instance_settings == settings
    assert instance_settings.model_fields_set == {"max_iter"}


def test_get_instance_settings_coercion():
    """Check that the options specific to an instance are validated and coerced."""
    algorithm_configuration = AlgorithmConfiguration(
        SLSQP_Settings(ftol_rel=0.5),
        instance_algorithm_options={
            "max_iter": lambda problem_configuration, index: str(index + 5)
        },
    )
    instance_settings = algorithm_configuration.get_instance_settings(mock.Mock(), 1)
    assert instance_settings.max_iter == 6
    assert instance_settings.ftol_rel == 0.5
    assert instance_settings.model_fields_set == {"max_iter", "ftol_rel"}


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_iters", 5, "max_iters"),
        ("max_iter", "five", "max_iter"),
    ],
)
def test_get_instance_settings_invalid_option(name, value, message):
    """Check that an invalid option specific to an instance raises an error."""
    algorithm_configuration = AlgorithmConfiguration(
        SLSQP_Settings(),
        instance_algorithm_options={name: lambda problem_configuration, index: value},
    )
    with pytest.raises(ValidationError, match=message):
        algorithm_configuration.get_instance_settings(mock.Mock(), 0)


def test_copy(algorithm_configuration):
    """Check the copy an algorithm configuration."""
    algorithm_configuration = AlgorithmConfiguration(
        Dummy_Settings(max_iter=9), "Algorithm configuration"
    ).copy()
    assert algorithm_configuration.algorithm_name == "Dummy"
    assert algorithm_configuration.name == "Algorithm configuration"
    assert algorithm_configuration.algorithm_options == {"max_iter": 9}
