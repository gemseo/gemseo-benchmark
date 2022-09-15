# Copyright 2022 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
"""A collection of algorithms configurations."""
from __future__ import annotations

import bisect
from typing import Iterator, MutableSet

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration


class AlgorithmsConfigurations(MutableSet[AlgorithmConfiguration]):
    """A collection of algorithms configurations."""

    def __init__(self, *algorithms_configurations: AlgorithmConfiguration) -> None:
        """
        Args:
            *algorithms_configurations: The algorithms configurations.
        """
        self.__algorithms = list()
        self.__configurations = list()
        self.__names = list()
        for configuration in algorithms_configurations:
            self.add(configuration)

    def __contains__(self, algorithm_configuration: AlgorithmConfiguration) -> bool:
        return algorithm_configuration.name in self.__names

    def __iter__(self) -> Iterator:
        return iter(self.__configurations)

    def __len__(self) -> int:
        return len(self.__configurations)

    def add(self, algorithm_configuration: AlgorithmConfiguration) -> None:
        """Add an algorithm configuration to the collection.

        Args:
            algorithm_configuration: The algorithm configuration.

        Raises:
            ValueError: If the collection already contains an algorithm configuration
                with the same name.
        """
        if algorithm_configuration in self:
            raise ValueError(
                "The collection already contains an algorithm configuration named "
                f"{algorithm_configuration.name}."
            )

        index = bisect.bisect(self.__names, algorithm_configuration.name)
        self.__configurations.insert(index, algorithm_configuration)
        bisect.insort(self.__names, algorithm_configuration.name)
        bisect.insort(self.__algorithms, algorithm_configuration.algorithm_name)

    def discard(self, algorithm_configuration: AlgorithmConfiguration) -> None:
        """Remove an algorithm configuration.

        Args:
            algorithm_configuration: The algorithm configuration to remove.
        """
        self.__configurations.remove(algorithm_configuration)
        self.__names.remove(algorithm_configuration.name)
        if algorithm_configuration.algorithm_name not in [
            algo_config.algorithm_name for algo_config in self
        ]:
            self.__algorithms.remove(algorithm_configuration.algorithm_name)

    @property
    def names(self) -> list[str]:
        """The names of the algorithms configurations in alphabetical order."""
        return self.__names

    @property
    def algorithms(self) -> list[str]:
        """The names of the algorithms in alphabetical order."""
        return self.__algorithms

    @property
    def configurations(self) -> list[AlgorithmConfiguration]:
        """The algorithms configurations."""
        return list(self.__configurations)