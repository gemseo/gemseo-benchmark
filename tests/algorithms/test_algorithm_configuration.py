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

from unittest import mock

import pytest
from gemseo.mda import MDAChain_Settings
from gemseo.mda import MDANewtonRaphson_Settings
from gemseo.optimization import MNBI_Settings
from gemseo.optimization import SLSQP_Settings
from gemseo.util.pydantic import BaseSettings
from gemseo.util.testing.helper import assert_exception
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


def test_name_several_options():
    """Check the name of an algorithm configuration with several options."""
    algorithm_configuration = AlgorithmConfiguration(
        SLSQP_Settings(max_iter=3, ftol_rel=0.1)
    )
    assert algorithm_configuration.name == "SLSQP_ftol_rel=0.1, max_iter=3"


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


@pytest.mark.parametrize(
    "dictionary",
    [
        {"configuration_name": "SLSQP", "algorithm_settings": {}},
        {},
        {
            "configuration_name": "SciPy SLSQP",
            "algorithm_name": "SLSQP",
            "algorithm_options": {"max_iter": 9},
        },
    ],
    ids=["one_key", "all_keys", "earlier_version"],
)
def test_from_dict_missing_keys(dictionary, snapshot):
    """Check the error message when an algorithm configuration misses keys.

    This is the case of an algorithm configuration saved by gemseo-benchmark 4.
    """
    with assert_exception(ValueError, snapshot):
        AlgorithmConfiguration.from_dict(dictionary)


@pytest.mark.parametrize(
    "settings_class",
    [
        "gemseo.algos.opt.mma.settings.mma_settings.MMASVanberg_Settings",
        "gemseo.optimization.scipy_local.settings.slsqp.Unknown_Settings",
    ],
)
def test_from_dict_unknown_settings_class(settings_class, snapshot):
    """Check the import of an algorithm configuration with an unknown settings class."""
    with assert_exception(ValueError, snapshot) as error:
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


@pytest.mark.parametrize(
    "instance_algorithm_options",
    [{}, {"max_iter": lambda problem_configuration, index: index + 5}],
    ids=["without_instance_options", "with_instance_options"],
)
def test_get_instance_settings_independent(instance_algorithm_options):
    """Check that each call returns new settings, independent of the original ones.

    The algorithm can change its settings while running,
    which is simulated by changing the returned settings.
    """
    settings = SLSQP_Settings(max_iter=9)
    algorithm_configuration = AlgorithmConfiguration(
        settings, instance_algorithm_options=instance_algorithm_options
    )
    first_settings = algorithm_configuration.get_instance_settings(mock.Mock(), 0)
    first_settings.ftol_rel = 0.5
    second_settings = algorithm_configuration.get_instance_settings(mock.Mock(), 0)
    assert first_settings is not settings
    assert second_settings is not settings
    assert second_settings is not first_settings
    assert second_settings.model_fields_set == {"max_iter"}
    assert second_settings.ftol_rel == settings.ftol_rel
    assert settings.model_dump(exclude_unset=True) == {"max_iter": 9}
    assert algorithm_configuration.algorithm_options == {"max_iter": 9}
    assert algorithm_configuration.name == "SLSQP_max_iter=9"


@pytest.mark.parametrize(
    "instance_algorithm_options",
    [{}, {"max_mda_iter": 3}],
    ids=["without_instance_options", "with_instance_options"],
)
def test_get_instance_settings_nested_settings(instance_algorithm_options):
    """Check that the nested settings keep their class and their fields.

    They are also copies of the original nested settings.
    """
    settings = MDAChain_Settings(
        inner_mda_settings=MDANewtonRaphson_Settings(over_relaxation_factor=0.5)
    )
    algorithm_configuration = AlgorithmConfiguration(
        settings,
        instance_algorithm_options={
            name: lambda problem_configuration, index, value=value: value
            for name, value in instance_algorithm_options.items()
        },
    )
    instance_settings = algorithm_configuration.get_instance_settings(mock.Mock(), 0)
    inner_mda_settings = instance_settings.inner_mda_settings
    assert isinstance(inner_mda_settings, MDANewtonRaphson_Settings)
    assert inner_mda_settings.over_relaxation_factor == 0.5
    assert inner_mda_settings is not settings.inner_mda_settings
    assert instance_settings.model_fields_set == {
        "inner_mda_settings",
        *instance_algorithm_options,
    }
    assert instance_settings.max_mda_iter == instance_algorithm_options.get(
        "max_mda_iter", settings.max_mda_iter
    )


def test_get_instance_settings_independent_nested_settings():
    """Check that the instances do not share their nested settings."""
    settings = MNBI_Settings(max_iter=100, sub_optim_algo_settings=SLSQP_Settings())
    algorithm_configuration = AlgorithmConfiguration(
        settings,
        instance_algorithm_options={
            "max_iter": lambda problem_configuration, index: 10 * (index + 1)
        },
    )
    first_settings = algorithm_configuration.get_instance_settings(mock.Mock(), 0)
    second_settings = algorithm_configuration.get_instance_settings(mock.Mock(), 1)
    first_sub_settings = first_settings.sub_optim_algo_settings
    second_sub_settings = second_settings.sub_optim_algo_settings
    assert (first_settings.max_iter, second_settings.max_iter) == (10, 20)
    assert isinstance(first_sub_settings, SLSQP_Settings)
    assert first_sub_settings is not second_sub_settings
    assert first_sub_settings is not settings.sub_optim_algo_settings
    first_sub_settings.ftol_rel = 0.5
    assert second_sub_settings.ftol_rel == settings.sub_optim_algo_settings.ftol_rel
    assert settings.sub_optim_algo_settings.ftol_rel != 0.5


class Other_Dummy_Settings(Dummy_Settings):  # ruff: ignore[invalid-class-name]
    """Settings of a fake algorithm with another default value."""

    max_iter: int = Field(default=7)
    ftol_rel: float = Field(default=0.1)


def test_get_instance_settings_subclass_defaults():
    """Check that the settings of a subclass keep their default values."""
    algorithm_configuration = AlgorithmConfiguration(
        Other_Dummy_Settings(ftol_rel=0.2),
        instance_algorithm_options={
            "ftol_rel": lambda problem_configuration, index: 0.3
        },
    )
    instance_settings = algorithm_configuration.get_instance_settings(mock.Mock(), 0)
    assert isinstance(instance_settings, Other_Dummy_Settings)
    assert instance_settings.max_iter == 7
    assert instance_settings.ftol_rel == 0.3
    assert instance_settings.model_fields_set == {"ftol_rel"}


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


@pytest.mark.parametrize(("name", "value"), [("max_iters", 5), ("max_iter", "five")])
def test_get_instance_settings_invalid_option(name, value, snapshot):
    """Check that an invalid option specific to an instance raises an error."""
    algorithm_configuration = AlgorithmConfiguration(
        SLSQP_Settings(),
        instance_algorithm_options={name: lambda problem_configuration, index: value},
    )
    with assert_exception(ValidationError, snapshot):
        algorithm_configuration.get_instance_settings(mock.Mock(), 0)


def test_copy(algorithm_configuration):
    """Check the copy an algorithm configuration."""
    algorithm_configuration = AlgorithmConfiguration(
        Dummy_Settings(max_iter=9), "Algorithm configuration"
    ).copy()
    assert algorithm_configuration.algorithm_name == "Dummy"
    assert algorithm_configuration.name == "Algorithm configuration"
    assert algorithm_configuration.algorithm_options == {"max_iter": 9}
