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

from math import inf
from typing import Any
from unittest import mock

import pytest
from gemseo.mda import MDAChain_Settings
from gemseo.mda import MDANewtonRaphson_Settings
from gemseo.optimization import MNBI_Settings
from gemseo.optimization import NELDER_MEAD_Settings
from gemseo.optimization import SLSQP_Settings
from gemseo.util.pydantic import BaseSettings
from gemseo.util.testing.helper import assert_exception
from numpy import ndarray
from numpy.testing import assert_equal
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithm_configuration import (
    _get_instance_settings_error_messages,
)
from gemseo_benchmark.algorithms.algorithm_configuration import _get_instances_settings


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
    ids=["missing_module", "missing_class"],
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


@pytest.mark.parametrize("settings_class", [None, 3], ids=["null", "integer"])
def test_from_dict_settings_class_not_a_string(settings_class, snapshot):
    """Check the import of an algorithm configuration with a malformed settings class.

    This is the case of a JSON file whose settings class is not a string.
    """
    with assert_exception(ValueError, snapshot):
        AlgorithmConfiguration.from_dict({
            "configuration_name": "MMA",
            "algorithm_settings_class": settings_class,
            "algorithm_settings": {},
        })


@pytest.mark.parametrize("settings", [None, ["max_iter", 9]], ids=["null", "list"])
def test_from_dict_settings_not_a_mapping(settings, snapshot):
    """Check the import of an algorithm configuration with malformed settings.

    This is the case of a JSON file whose algorithm settings are not an object.
    """
    with assert_exception(ValueError, snapshot):
        AlgorithmConfiguration.from_dict({
            "configuration_name": "SLSQP",
            "algorithm_settings_class": "gemseo.optimization.SLSQP_Settings",
            "algorithm_settings": settings,
        })


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


def raise_index_error(problem_configuration, index):
    """Raise an `IndexError`.

    Args:
        problem_configuration: The problem configuration.
        index: The 0-based index of the problem.

    Raises:
        IndexError: Always.
    """
    msg = "There is no value for this problem."
    raise IndexError(msg)


@pytest.mark.parametrize(
    ("name", "get_value", "cause"),
    [
        ("max_iters", lambda problem_configuration, index: 5, ValidationError),
        ("max_iter", lambda problem_configuration, index: "five", ValidationError),
        ("max_iter", raise_index_error, IndexError),
    ],
    ids=["unknown_option", "invalid_value", "callable_error"],
)
def test_get_instance_settings_invalid_option(name, get_value, cause, snapshot):
    """Check that an invalid option specific to an instance raises an error."""
    algorithm_configuration = AlgorithmConfiguration(
        SLSQP_Settings(), "SLSQP", instance_algorithm_options={name: get_value}
    )
    problem_configuration = mock.Mock()
    problem_configuration.name = "Rosenbrock"
    with assert_exception(ValueError, snapshot) as error:
        algorithm_configuration.get_instance_settings(problem_configuration, 1)

    assert isinstance(error.value.__cause__, cause)


class _Prefixed_Settings(Dummy_Settings):  # ruff: ignore[invalid-class-name]
    """Settings of a fake algorithm whose option changes at each validation."""

    label: str = Field(default="")

    @field_validator("label", mode="before")
    @classmethod
    def prefix_label(cls, value: Any) -> str:
        """Prefix the label by its type.

        Args:
            value: The label as given.

        Returns:
            The label prefixed by its type.

        Raises:
            TypeError: When the label is not a string.
        """
        if not isinstance(value, str):
            msg = "The label must be a string."
            raise TypeError(msg)

        return f"label {value}"


def test_get_instance_settings_other_values_kept():
    """Check that the options not specific to an instance keep their values.

    These are the values validated when the settings were created,
    even if validating them again would change them.
    """
    settings = _Prefixed_Settings(label="1")
    algorithm_configuration = AlgorithmConfiguration(
        settings,
        instance_algorithm_options={
            "max_iter": lambda problem_configuration, index: index + 5
        },
    )
    instance_settings = algorithm_configuration.get_instance_settings(mock.Mock(), 0)
    assert instance_settings.label == "label 1"
    assert instance_settings.max_iter == 5
    assert instance_settings.model_fields_set == {"label", "max_iter"}
    assert settings.max_iter == 100


def test_get_instance_settings_initial_simplex():
    """Check that the model validators coerce the options specific to an instance.

    The settings of the Nelder-Mead algorithm cast the initial simplex
    as a NumPy array before validating the fields.
    """
    algorithm_configuration = AlgorithmConfiguration(
        NELDER_MEAD_Settings(max_iter=9),
        instance_algorithm_options={
            "initial_simplex": lambda problem_configuration, index: [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        },
    )
    instance_settings = algorithm_configuration.get_instance_settings(mock.Mock(), 0)
    assert isinstance(instance_settings.initial_simplex, ndarray)
    assert_equal(instance_settings.initial_simplex, [[0, 0], [1, 0], [0, 1]])
    assert instance_settings.max_iter == 9


@pytest.mark.parametrize(
    "names",
    [("store_jacobian", "kkt_tol_abs"), ("kkt_tol_abs", "store_jacobian")],
    ids=["store_jacobian_first", "kkt_tol_abs_first"],
)
def test_get_instance_settings_options_checked_together(names):
    """Check that the options are checked together, whatever their order.

    The KKT options can be set only if the Jacobian is stored;
    an option specific to an instance unsets the KKT option set by the user
    and another one stops storing the Jacobian.
    """
    values = {"store_jacobian": False, "kkt_tol_abs": inf}
    algorithm_configuration = AlgorithmConfiguration(
        SLSQP_Settings(kkt_tol_abs=1.0),
        instance_algorithm_options={
            name: lambda problem_configuration, index, value=values[name]: value
            for name in names
        },
    )
    instance_settings = algorithm_configuration.get_instance_settings(mock.Mock(), 0)
    assert not instance_settings.store_jacobian
    assert instance_settings.kkt_tol_abs == inf


@pytest.mark.parametrize(
    ("errors", "messages"),
    [
        (
            [("A", "P", 1, ValueError("Wrong."))],
            [
                (
                    "The options of algorithm configuration A "
                    "for problem 2 of problem configuration P are invalid: Wrong."
                )
            ],
        ),
        (
            [("A", "P", index, ValueError("Wrong.")) for index in range(3)],
            [
                (
                    "The options of algorithm configuration A "
                    "for problems 1, 2 and 3 of problem configuration P "
                    "are invalid: Wrong."
                )
            ],
        ),
        (
            [
                ("A", "P", 0, ValueError("Wrong.")),
                ("A", "Q", 1, ValueError("Wrong.")),
            ],
            [
                (
                    "The options of algorithm configuration A "
                    "for problem 1 of problem configuration P "
                    "and problem 2 of problem configuration Q are invalid: Wrong."
                )
            ],
        ),
        (
            [
                ("A", "P", 0, ValueError("Wrong.")),
                ("A", "Q", 0, ValueError("Wrong.")),
                ("A", "Q", 1, ValueError("Wrong.")),
                ("A", "R", 2, ValueError("Wrong.")),
            ],
            [
                (
                    "The options of algorithm configuration A "
                    "for problem 1 of problem configuration P, "
                    "problems 1 and 2 of problem configuration Q "
                    "and problem 3 of problem configuration R are invalid: Wrong."
                )
            ],
        ),
        (
            [
                ("A", "P", 0, ValueError("Wrong.")),
                ("A", "P", 1, ValueError("Other.")),
                ("A", "P", 2, ValueError("Wrong.")),
            ],
            [
                (
                    "The options of algorithm configuration A "
                    "for problems 1 and 3 of problem configuration P "
                    "are invalid: Wrong."
                ),
                (
                    "The options of algorithm configuration A "
                    "for problem 2 of problem configuration P are invalid: Other."
                ),
            ],
        ),
        (
            [
                ("A", "P", 0, ValueError("Wrong.")),
                ("B", "P", 0, ValueError("Wrong.")),
            ],
            [
                (
                    "The options of algorithm configuration A "
                    "for problem 1 of problem configuration P are invalid: Wrong."
                ),
                (
                    "The options of algorithm configuration B "
                    "for problem 1 of problem configuration P are invalid: Wrong."
                ),
            ],
        ),
    ],
    ids=[
        "one_instance",
        "same_error_one_problem_configuration",
        "same_error_two_problem_configurations",
        "same_error_three_problem_configurations",
        "different_errors",
        "same_error_several_algorithm_configurations",
    ],
)
def test_get_instance_settings_error_messages(errors, messages):
    """Check that the instances sharing an error are named in one message."""
    assert _get_instance_settings_error_messages(errors) == messages


def test_get_instance_settings_without_instance_options():
    """Check that the settings are not validated again without instance options."""
    settings = SLSQP_Settings(max_iter=9)
    algorithm_configuration = AlgorithmConfiguration(settings)
    with mock.patch.object(
        SLSQP_Settings, "model_validate", wraps=SLSQP_Settings.model_validate
    ) as model_validate:
        instance_settings = algorithm_configuration.get_instance_settings(
            mock.Mock(), 0
        )

    model_validate.assert_not_called()
    assert instance_settings is not settings
    assert instance_settings == settings
    assert instance_settings.model_fields_set == {"max_iter"}


def test_get_instances_settings():
    """Check the settings of algorithms for instances of problems.

    The executions whose settings are invalid have no settings,
    and the errors are grouped in the order of their first occurrence.
    """
    valid_configuration = AlgorithmConfiguration(SLSQP_Settings(max_iter=9), "A")
    invalid_configuration = AlgorithmConfiguration(
        SLSQP_Settings(),
        "B",
        instance_algorithm_options={"max_iter": lambda problem, index: "five"},
    )
    problem_configuration = mock.Mock()
    problem_configuration.name = "P"
    settings, messages = _get_instances_settings(
        [
            (invalid_configuration, problem_configuration, 0),
            (valid_configuration, problem_configuration, 0),
            (invalid_configuration, problem_configuration, 1),
        ],
        copy_settings_without_instance_options=True,
    )
    assert settings[0] is None
    assert settings[1].max_iter == 9
    assert settings[2] is None
    assert len(messages) == 1
    assert messages[0].startswith(
        "The options of algorithm configuration B "
        "for problems 1 and 2 of problem configuration P are invalid: "
    )


@pytest.mark.parametrize(
    ("copy_settings", "with_instance_options", "shared"),
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (True, True, False),
    ],
    ids=[
        "no_copy_without_instance_options",
        "copy_without_instance_options",
        "no_copy_with_instance_options",
        "copy_with_instance_options",
    ],
)
def test_get_instances_settings_copy(copy_settings, with_instance_options, shared):
    """Check when the executions share the settings of the algorithm configuration.

    Only an algorithm configuration without options specific to the instances
    shares its settings, and only when they are not to be copied.
    """
    algorithm_settings = SLSQP_Settings(max_iter=9)
    instance_algorithm_options = (
        {"max_iter": lambda problem, index: index + 5} if with_instance_options else {}
    )
    algorithm_configuration = AlgorithmConfiguration(
        algorithm_settings, instance_algorithm_options=instance_algorithm_options
    )
    problem_configuration = mock.Mock()
    problem_configuration.name = "P"
    with mock.patch.object(
        SLSQP_Settings, "model_validate", wraps=SLSQP_Settings.model_validate
    ) as model_validate:
        settings, messages = _get_instances_settings(
            [
                (algorithm_configuration, problem_configuration, 0),
                (algorithm_configuration, problem_configuration, 1),
            ],
            copy_settings_without_instance_options=copy_settings,
        )

    assert not messages
    assert (settings[0] is algorithm_settings) is shared
    assert (settings[1] is algorithm_settings) is shared
    assert (settings[0] is settings[1]) is shared
    assert model_validate.called is with_instance_options


def test_get_instances_settings_error_without_cause():
    """Check the message of an error raised without a cause."""
    algorithm_configuration = mock.Mock(
        get_instance_settings=mock.Mock(side_effect=ValueError("Wrong.")),
        instance_algorithm_options={"max_iter": mock.Mock()},
    )
    algorithm_configuration.name = "A"
    problem_configuration = mock.Mock()
    problem_configuration.name = "P"
    assert _get_instances_settings(
        [(algorithm_configuration, problem_configuration, 0)],
        copy_settings_without_instance_options=True,
    ) == (
        [None],
        [
            (
                "The options of algorithm configuration A "
                "for problem 1 of problem configuration P are invalid: Wrong."
            )
        ],
    )


def test_copy(algorithm_configuration):
    """Check the copy an algorithm configuration."""
    algorithm_configuration = AlgorithmConfiguration(
        Dummy_Settings(max_iter=9), "Algorithm configuration"
    ).copy()
    assert algorithm_configuration.algorithm_name == "Dummy"
    assert algorithm_configuration.name == "Algorithm configuration"
    assert algorithm_configuration.algorithm_options == {"max_iter": 9}
