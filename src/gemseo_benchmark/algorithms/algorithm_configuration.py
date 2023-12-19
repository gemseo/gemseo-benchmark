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
"""Configuration of an algorithm defined by the values of its options.

An algorithm depends on the values of its options. A value set defines a configuration
of the algorithm.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from types import MappingProxyType
from typing import Any
from typing import Callable
from typing import Final

from gemseo.utils.string_tools import pretty_repr

InstanceAlgorithmOptions = MutableMapping[str, Callable[[int], Any]]


class AlgorithmConfiguration:
    """The configuration of an algorithm."""

    __ALGORITHM_NAME: Final[str] = "algorithm_name"
    __ALGORITHM_OPTIONS: Final[str] = "algorithm_options"
    __CONFIGURATION_NAME: Final[str] = "configuration_name"
    __INSTANCE_ALGORITHM_OPTIONS: Final[str] = "instance_algorithm_options"

    def __init__(
        self,
        algorithm_name: str,
        configuration_name: str | None = None,
        instance_algorithm_options: InstanceAlgorithmOptions = MappingProxyType({}),
        **algorithm_options: Any,
    ) -> None:
        """
        Args:
            algorithm_name: The name of the algorithm.
            configuration_name: The name of the configuration of the algorithm.
                If ``None``, a name will be generated based on the algorithm name and
                its options, based on the pattern
                ``"algorithm_name[option_name=option_value, ...]"``.
            instance_algorithm_options: The options of the algorithm specific to
                instances of a problem.
                They shall be passed as a mapping
                that links the name of an algorithm option
                to a callable that takes the 0-based index of the instance as argument
                and returns the value of the option.
            **algorithm_options: The options of the algorithm.
        """  # noqa: D205, D212, D415
        self.__algorithm_name = algorithm_name
        self.__algorithm_options = algorithm_options
        self.__configuration_name = configuration_name or self.__get_configuration_name(
            algorithm_name, **algorithm_options
        )
        self.__instance_algorithm_options = instance_algorithm_options

    @staticmethod
    def __get_configuration_name(algorithm_name: str, **algorithm_options: Any) -> str:
        """Define a name for the configuration based on the algorithm name and options.

        Args:
            algorithm_name: The name of the algorithm.
            **algorithm_options: The options of the algorithm.

        Returns:
            The name of the algorithm configuration.
        """
        if not algorithm_options:
            return algorithm_name

        return f"{algorithm_name}_{pretty_repr(algorithm_options)}"

    @property
    def name(self) -> str:
        """The name of the algorithm configuration."""
        return self.__configuration_name

    @property
    def algorithm_name(self) -> str:
        """The name of the algorithm."""
        return self.__algorithm_name

    @property
    def algorithm_options(self) -> dict[str, Any]:
        """The options of the algorithm."""
        return self.__algorithm_options

    @property
    def instance_algorithm_options(self) -> InstanceAlgorithmOptions:
        """The instance-specific options of the algorithm."""
        return self.__instance_algorithm_options

    def to_dict(
        self, skip_instance_algorithm_options: bool = False
    ) -> dict[str, str | dict[str, Any]]:
        """Return the algorithm configuration as a dictionary.

        Args:
            skip_instance_algorithm_options: Whether to skip the algorithm options
                specific to problem instances.

        Returns:
            The algorithm configuration as a dictionary.
        """
        dictionary = {
            self.__CONFIGURATION_NAME: self.__configuration_name,
            self.__ALGORITHM_NAME: self.__algorithm_name,
            self.__ALGORITHM_OPTIONS: self.__algorithm_options,
        }
        if not skip_instance_algorithm_options:
            dictionary[self.__INSTANCE_ALGORITHM_OPTIONS] = (
                self.__instance_algorithm_options
            )

        return dictionary

    @classmethod
    def from_dict(
        cls, algorithm_configuration: dict[str, str | dict[str, Any]]
    ) -> AlgorithmConfiguration:
        """Load an algorithm configuration from a dictionary.

        Args:
            algorithm_configuration: The algorithm configuration.

        Returns:
            The algorithm configuration.
        """
        return AlgorithmConfiguration(
            algorithm_configuration[cls.__ALGORITHM_NAME],
            algorithm_configuration[cls.__CONFIGURATION_NAME],
            algorithm_configuration.get(cls.__INSTANCE_ALGORITHM_OPTIONS, {}),
            **algorithm_configuration[cls.__ALGORITHM_OPTIONS],
        )

    def copy(self) -> AlgorithmConfiguration:
        """Return a copy of the algorithm configuration.

        Returns:
            A copy of the algorithm configuration.
        """
        return AlgorithmConfiguration(
            self.algorithm_name,
            self.name,
            self.instance_algorithm_options,
            **self.algorithm_options,
        )
