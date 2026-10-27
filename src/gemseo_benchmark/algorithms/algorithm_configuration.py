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
"""Configuration of an algorithm defined by the values of its options."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from gemseo.util.constant import read_only_empty_dict
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.util.pydantic import BaseSettings

    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )

InstanceAlgorithmOptions = MutableMapping[
    str, Callable[["BaseProblemConfiguration", int], Any]
]
"""The options of an algorithm specific to the instances of a problem.

The mapping links the name of an algorithm option to a callable
that takes the problem configuration and the 0-based index of the instance
and returns the value of the option.
"""


class AlgorithmConfiguration:
    """The configuration of an algorithm.

    An algorithm depends on the values of its options.
    A value set defines a configuration of the algorithm.
    """

    __ALGORITHM_SETTINGS_CLASS: Final[str] = "algorithm_settings_class"
    __ALGORITHM_SETTINGS: Final[str] = "algorithm_settings"
    __CONFIGURATION_NAME: Final[str] = "configuration_name"
    __INSTANCE_ALGORITHM_OPTIONS: Final[str] = "instance_algorithm_options"

    def __init__(
        self,
        algorithm_settings: BaseSettings,
        configuration_name: str = "",
        instance_algorithm_options: InstanceAlgorithmOptions = read_only_empty_dict,
    ) -> None:
        """
        Args:
            algorithm_settings: The settings of the algorithm.
            configuration_name: The name of the configuration of the algorithm.
                If empty, a name will be generated based on the algorithm name and
                its options, based on the pattern
                `"algorithm_name[option_name=option_value, ...]"`.
            instance_algorithm_options: The options of the algorithm specific to
                instances of a problem.
                They shall be passed as a mapping
                that links the name of an algorithm option
                to a callable that takes the problem configuration
                and the 0-based index of the instance as arguments
                and returns the value of the option.
        """  # ruff: ignore[missing-blank-line-after-summary, multi-line-summary-first-line, missing-terminal-punctuation]
        self.__algorithm_settings = algorithm_settings
        algorithm_name = algorithm_settings.target_class_name
        non_default_opts = algorithm_settings.model_dump(exclude_unset=True)
        self.__configuration_name = configuration_name or self.__get_configuration_name(
            algorithm_name, non_default_opts
        )
        self.__instance_algorithm_options = instance_algorithm_options

    @classmethod
    def __get_configuration_name(
        cls, algorithm_name: str, options: dict[str, Any]
    ) -> str:
        """Return a name for the configuration based on the algorithm name and options.

        Args:
            algorithm_name: The name of the algorithm.
            options: The explicitly-set options of the algorithm.

        Returns:
            The name of the algorithm configuration.
        """
        if not options:
            return algorithm_name

        options = pretty_repr(cls.__make_json_serializable(options), use_and=False)
        return f"{algorithm_name}_{options}"

    @property
    def name(self) -> str:
        """The name of the algorithm configuration."""
        return self.__configuration_name

    @property
    def algorithm_settings(self) -> BaseSettings:
        """The settings of the algorithm."""
        return self.__algorithm_settings

    @property
    def algorithm_name(self) -> str:
        """The name of the algorithm."""
        return self.__algorithm_settings.target_class_name

    @property
    def algorithm_options(self) -> dict[str, Any]:
        """The explicitly-set options of the algorithm."""
        return self.__algorithm_settings.model_dump(exclude_unset=True)

    @property
    def instance_algorithm_options(self) -> InstanceAlgorithmOptions:
        """The instance-specific options of the algorithm."""
        return self.__instance_algorithm_options

    def get_instance_settings(
        self, problem_configuration: BaseProblemConfiguration, index: int
    ) -> BaseSettings:
        """Return the settings of the algorithm for an instance of a problem.

        Args:
            problem_configuration: The problem configuration.
            index: The 0-based index of the instance of the problem.

        Returns:
            A new copy of the settings of the algorithm,
            nested settings included,
            updated with the options specific to the instance of the problem.
            Without such options, the copy is not validated again.

        Raises:
            ValidationError: When an option specific to the instance
                is not an option of the algorithm or has an invalid value.
        """
        # The algorithm can change its settings while running,
        # so that each execution needs its own settings.
        settings = self.__algorithm_settings.model_copy(deep=True)
        if not self.__instance_algorithm_options:
            return settings

        # The values of the explicitly-set fields are passed as they are,
        # so that the nested settings keep their class;
        # they are copies, so that the instances do not share them.
        return type(settings).model_validate({
            **{name: getattr(settings, name) for name in settings.model_fields_set},
            **{
                name: function(problem_configuration, index)
                for name, function in self.__instance_algorithm_options.items()
            },
        })

    def get_instance_settings_error_message(
        self,
        problem_configuration: BaseProblemConfiguration,
        index: int,
        error: Exception,
    ) -> str:
        """Return the message of an error raised by the settings of an instance.

        Args:
            problem_configuration: The problem configuration.
            index: The 0-based index of the instance of the problem.
            error: The error raised by `get_instance_settings`.

        Returns:
            The message of the error,
            naming the algorithm configuration and the instance of the problem.
        """
        return (
            f"The options of algorithm configuration {self.__configuration_name} "
            f"for problem {index + 1} "
            f"of problem configuration {problem_configuration.name} "
            f"are invalid: {error}"
        )

    def to_dict(self, skip_instance_algorithm_options: bool = False) -> dict[str, Any]:
        """Return the algorithm configuration as a dictionary.

        Args:
            skip_instance_algorithm_options: Whether to skip the algorithm options
                specific to problem instances.

        Returns:
            The algorithm configuration as a dictionary.
        """
        settings_class = type(self.__algorithm_settings)
        dictionary = {
            self.__CONFIGURATION_NAME: self.__configuration_name,
            self.__ALGORITHM_SETTINGS_CLASS: (
                f"{settings_class.__module__}.{settings_class.__qualname__}"
            ),
            self.__ALGORITHM_SETTINGS: self.__make_json_serializable(
                self.algorithm_options
            ),
        }
        if not skip_instance_algorithm_options:
            dictionary[self.__INSTANCE_ALGORITHM_OPTIONS] = (
                self.__instance_algorithm_options
            )

        return dictionary

    @staticmethod
    def __make_json_serializable(
        data: Mapping,
    ) -> dict[str, bool | int | float | tuple | list | dict | None]:
        """Make data JSON-serializable.

        Args:
            data: The data.

        Returns:
            The JSON-serializable data.
        """
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in data.items()
        }

    @classmethod
    def from_dict(
        cls, algorithm_configuration: dict[str, Any]
    ) -> AlgorithmConfiguration:
        """Load an algorithm configuration from a dictionary.

        Args:
            algorithm_configuration: The algorithm configuration.

        Returns:
            The algorithm configuration.

        Raises:
            ValueError: If the dictionary misses a key
                or if the class of the algorithm settings cannot be imported,
                e.g. because the plugin providing the algorithm is not installed
                or because it was written by an earlier version of gemseo-benchmark.
        """
        missing_keys = [
            key
            for key in (
                cls.__CONFIGURATION_NAME,
                cls.__ALGORITHM_SETTINGS_CLASS,
                cls.__ALGORITHM_SETTINGS,
            )
            if key not in algorithm_configuration
        ]
        if missing_keys:
            word = "key" if len(missing_keys) == 1 else "keys"
            msg = (
                "The algorithm configuration misses the "
                f"{word} {pretty_repr(missing_keys, sort=False)}; "
                "it was probably written by an earlier version of gemseo-benchmark "
                "and must be computed again."
            )
            raise ValueError(msg)

        fqn = algorithm_configuration[cls.__ALGORITHM_SETTINGS_CLASS]
        module_name, _, class_name = fqn.rpartition(".")
        try:
            settings_class = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError, ValueError) as error:
            msg = (
                f"The class of the algorithm settings {fqn!r} cannot be imported; "
                "install the plugin providing the algorithm, "
                "or, if the results were written by an earlier version "
                "of gemseo-benchmark, compute them again."
            )
            raise ValueError(msg) from error

        settings = settings_class(**algorithm_configuration[cls.__ALGORITHM_SETTINGS])
        return cls(
            settings,
            algorithm_configuration[cls.__CONFIGURATION_NAME],
            algorithm_configuration.get(cls.__INSTANCE_ALGORITHM_OPTIONS, {}),
        )

    def copy(self) -> AlgorithmConfiguration:
        """Return a copy of the algorithm configuration.

        Returns:
            A copy of the algorithm configuration.
        """
        return AlgorithmConfiguration(
            self.__algorithm_settings.model_copy(),
            self.name,
            dict(self.instance_algorithm_options),
        )
