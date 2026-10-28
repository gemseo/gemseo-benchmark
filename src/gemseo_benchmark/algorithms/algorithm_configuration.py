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
from collections.abc import Mapping
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from gemseo.util.constant import read_only_empty_dict
from gemseo.util.pydantic import BaseSettings
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

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


def _get_instances_settings(
    executions: Iterable[tuple[AlgorithmConfiguration, BaseProblemConfiguration, int]],
    *,
    copy_settings_without_instance_options: bool,
) -> tuple[list[BaseSettings | None], list[str]]:
    """Return the settings of algorithms for instances of problems.

    The settings of all the executions are computed
    before reporting any error,
    so that all the mistakes can be reported together.

    Args:
        executions: The executions,
            as tuples of the form
            `(algorithm configuration, problem configuration,
            0-based index of the instance)`.
        copy_settings_without_instance_options: Whether an algorithm configuration
            without options specific to the instances
            gets a new copy of its settings for each execution.
            Otherwise, its executions share its settings,
            which must then be copied before running the algorithm,
            as the algorithm can change its settings while running.

    Returns:
        The settings of the algorithms, one per execution and in the same order,
        `None` for an execution whose settings are invalid,
        and the messages of the errors,
        one per algorithm configuration and error,
        in the order of the first occurrence of each error.
    """
    settings = []
    errors = []
    for algorithm_configuration, problem_configuration, index in executions:
        if (
            not copy_settings_without_instance_options
            and not algorithm_configuration.instance_algorithm_options
        ):
            settings.append(algorithm_configuration.algorithm_settings)
            continue

        try:
            settings.append(
                algorithm_configuration._get_instance_settings(
                    problem_configuration, index
                )
            )
        except Exception as error:  # ruff: ignore[blind-except]
            # Any error is reported,
            # as a callable defining an option specific to the problems
            # or a validator of the settings can raise any exception.
            settings.append(None)
            errors.append((
                algorithm_configuration.name,
                problem_configuration.name,
                index,
                error,
            ))

    return settings, _get_instance_settings_error_messages(errors)


def _get_instance_settings_error_messages(
    errors: Iterable[tuple[str, str, int, BaseException]],
) -> list[str]:
    """Return one message per algorithm configuration and error.

    The instances of the problems sharing the same error
    for the same algorithm configuration are named in the same message.

    Args:
        errors: The errors raised when computing the settings of algorithms
            for instances of problems,
            as tuples of the form
            `(algorithm configuration name, problem configuration name,
            0-based index of the instance, error)`.

    Returns:
        The messages, in the order of the first occurrence of each error.
    """
    # The 1-based indices of the instances,
    # per problem configuration name,
    # per algorithm configuration name and error message.
    instances: dict[tuple[str, str], dict[str, list[int]]] = {}
    for configuration_name, problem_name, index, error in errors:
        instances.setdefault((configuration_name, str(error)), {}).setdefault(
            problem_name, []
        ).append(index + 1)

    return [
        f"The options of algorithm configuration {configuration_name} "
        f"for {_describe_instances(problems)} are invalid: {message}"
        for (configuration_name, message), problems in instances.items()
    ]


def _describe_instances(instances: Mapping[str, Sequence[int]]) -> str:
    """Describe instances of problems.

    Args:
        instances: The 1-based indices of the instances
            per problem configuration name.

    Returns:
        The description of the instances,
        e.g. `"problems 1 and 2 of problem configuration Rosenbrock"`.
    """
    descriptions = [
        f"problem{'s' if len(indices) > 1 else ''} {pretty_repr(indices, sort=False)} "
        f"of problem configuration {problem_name}"
        for problem_name, indices in instances.items()
    ]
    if len(descriptions) == 1:
        return descriptions[0]

    return f"{', '.join(descriptions[:-1])} and {descriptions[-1]}"


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
        self.__configuration_name = configuration_name or self.__get_configuration_name(
            algorithm_settings.target_class_name,
            algorithm_settings.model_dump(exclude_unset=True),
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
            Without such options, the settings are not validated again.
            Otherwise, all the explicitly-set options are validated together,
            as when creating the settings,
            so that the validators of the settings see the final options;
            the options not specific to the instance keep their values,
            which their validators must accept as input.

        Raises:
            ValueError: When an option specific to the instance
                is not an option of the algorithm or has an invalid value,
                or when the options are not valid together,
                or when the callable computing it raises an error;
                the message names the algorithm configuration
                and the instance of the problem.
        """
        try:
            return self._get_instance_settings(problem_configuration, index)
        except Exception as error:  # ruff: ignore[blind-except]
            # Any error is reported,
            # as a callable defining an option specific to the problems
            # or a validator of the settings can raise any exception.
            msg = _get_instance_settings_error_messages([
                (self.__configuration_name, problem_configuration.name, index, error)
            ])[0]
            raise ValueError(msg) from error

    def _get_instance_settings(
        self, problem_configuration: BaseProblemConfiguration, index: int
    ) -> BaseSettings:
        """Return the settings of the algorithm for an instance of a problem.

        Args:
            problem_configuration: The problem configuration.
            index: The 0-based index of the instance of the problem.

        Returns:
            The settings of the algorithm for the instance of the problem,
            as returned by `get_instance_settings`.

        Raises:
            Exception: Any error raised by a callable computing an option
                specific to the instance
                or by the validation of the settings,
                without a message naming the algorithm configuration
                and the instance of the problem.
        """
        # The algorithm can change its settings while running,
        # so that each execution needs its own settings.
        settings = self.__algorithm_settings.model_copy(deep=True)
        if not self.__instance_algorithm_options:
            return settings

        options = {name: getattr(settings, name) for name in settings.model_fields_set}
        instance_options = {
            name: function(problem_configuration, index)
            for name, function in self.__instance_algorithm_options.items()
        }
        # One validation of all the options,
        # as when creating the settings,
        # so that the validators of the settings see the final options.
        instance_settings = type(settings).model_validate({
            **options,
            **instance_options,
        })
        # The other options keep the values validated when the settings were created,
        # even if validating them again changed them.
        return instance_settings.model_copy(
            update={
                name: value
                for name, value in options.items()
                if name not in instance_options
            }
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
            ValueError: If the dictionary misses a key,
                if the class of the algorithm settings is not a string,
                if it cannot be imported
                or if it is not a class of settings of a GEMSEO algorithm,
                e.g. because the plugin providing the algorithm is not installed
                or because it was written by an earlier version of gemseo-benchmark,
                or if the algorithm settings are not a mapping.
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
        if not isinstance(fqn, str):
            msg = (
                f"The class of the algorithm settings {fqn!r} is not a string; "
                "it must be the fully qualified name of a class, "
                "e.g. 'gemseo.optimization.SLSQP_Settings'."
            )
            # A ValueError, like the other errors of a malformed configuration,
            # so that the readers of the files can name the file.
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]

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

        # The class is checked before calling it,
        # so that the only object of the file that is called
        # is a class of settings of a GEMSEO algorithm.
        # This does not prevent the import of the module named in the file,
        # which runs the top-level code of this module;
        # so, the results files must come from a trusted source.
        if not (
            isinstance(settings_class, type)
            and issubclass(settings_class, BaseSettings)
        ):
            msg = (
                f"The class of the algorithm settings {fqn!r} "
                "is not a class of settings of a GEMSEO algorithm; "
                "it must be the fully qualified name of such a class, "
                "e.g. 'gemseo.optimization.SLSQP_Settings'."
            )
            # A ValueError, like the other errors of a malformed configuration,
            # so that the readers of the files can name the file.
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]

        options = algorithm_configuration[cls.__ALGORITHM_SETTINGS]
        if not isinstance(options, Mapping):
            msg = (
                f"The algorithm settings {options!r} are not a mapping "
                "from the names of the options to their values; "
                "if the results were written by an earlier version "
                "of gemseo-benchmark, compute them again."
            )
            # A ValueError, like the other errors of a malformed configuration,
            # so that the readers of the files can name the file.
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]

        settings = settings_class(**options)
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
