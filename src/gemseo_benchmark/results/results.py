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
"""A class to collect the paths to performance histories."""

from __future__ import annotations

import json
from pathlib import Path


class Results:
    """A collection of paths to performance histories."""

    def __init__(self, path: str | Path | None = None) -> None:
        """
        Args:
            path: The path to the JSON file from which to load the paths.
                If ``None``, the collection is initially empty.
        """  # noqa: D205, D212, D415
        self.__dict = {}
        if path is not None:
            self.from_file(path)

    def add_path(
        self, algorithm_configuration_name: str, problem_name: str, path: str | Path
    ) -> None:
        """Add a path to a performance history.

        Args:
            algorithm_configuration_name: The name of the algorithm configuration
                associated with the history.
            problem_name: The name of the problem associated with the history.
            path: The path to the history.

        Raises:
            FileNotFoundError: If the path to the history does not exist.
        """
        try:
            absolute_path = Path(path).resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The path to the history does not exist: {path}."
            ) from None
        if algorithm_configuration_name not in self.__dict:
            self.__dict[algorithm_configuration_name] = {}

        if problem_name not in self.__dict[algorithm_configuration_name]:
            self.__dict[algorithm_configuration_name][problem_name] = []

        self.__dict[algorithm_configuration_name][problem_name].append(absolute_path)

    def to_file(self, path: str | Path, indent: int | None = None) -> None:
        """Save the histories paths to a JSON file.

        Args:
            path: The path where to save the JSON file.
            indent: The indent level of the JSON serialization.
        """
        # Convert the paths to strings to be JSON serializable
        serializable = {}
        for algo_name, problems in self.__dict.items():
            serializable[algo_name] = {}
            for problem_name, paths in problems.items():
                serializable[algo_name][problem_name] = [str(path) for path in paths]
        with Path(path).open("w") as file:
            json.dump(serializable, file, indent=indent)

    def from_file(self, path: str | Path) -> None:
        """Load paths to performance histories from a JSON file.

        Args:
            path: The path to the JSON file.
        """
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"The path to the JSON file does not exist: {path}."
            )

        with Path(path).open("r") as file:
            histories = json.load(file)
        for algo_name, problems in histories.items():
            for problem_name, paths in problems.items():
                for path in paths:
                    self.add_path(algo_name, problem_name, path)

    @property
    def algorithms(self) -> list[str]:
        """Return the names of the algorithms configurations.

        Returns:
            The names of the algorithms configurations.
        """
        return list(self.__dict)

    def get_problems(self, algo_name: str) -> list[str]:
        """Return the names of the problems for a given algorithm configuration.

        Args:
            algo_name: The name of the algorithm configuration.

        Returns:
            The names of the problems.

        Raises:
            ValueError: If the algorithm configuration name is unknown.
        """
        if algo_name not in self.__dict:
            raise ValueError(f"Unknown algorithm name: {algo_name}.")

        return list(self.__dict[algo_name])

    def get_paths(self, algo_name: str, problem_name: str) -> list[Path]:
        """Return the paths associated with an algorithm and a problem.

        Args:
            algo_name: The name of the algorithm.
            problem_name: The name of the problem.

        Returns:
            The paths to the performance histories.

        Raises:
            ValueError: If the algorithm name is unknown,
                or if the problem name is unknown.
        """
        if algo_name not in self.__dict:
            raise ValueError(f"Unknown algorithm name: {algo_name}.")

        if problem_name not in self.__dict[algo_name]:
            raise ValueError(f"Unknown problem name: {problem_name}.")

        return self.__dict[algo_name][problem_name]

    def contains(self, algo_name: str, problem_name: str, path: Path) -> bool:
        """Check whether a result is stored.

        Args:
            algo_name: The name of the algorithm configuration.
            problem_name: The name of the problem.
            path: The path to the performance history

        Returns:
            Whether the result is stored.
        """
        return (
            algo_name in self.__dict
            and problem_name in self.__dict[algo_name]
            and path in self.__dict[algo_name][problem_name]
        )
