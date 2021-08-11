# -*- coding: utf-8 -*-
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
import json

from gemseo.utils.py23_compat import Path


class Results(object):
    """A collection of paths to performance histories."""

    def __init__(
            self,
            path=None,  # type: Optional[Union[str, Path]]
    ):  # type: (...) -> None
        """
        Args:
            path: The path to the JSON file from which to load the paths.
                If None, the collection is initially empty.
        """
        self.__dict = dict()
        if path is not None:
            self.from_file(path)

    def add_path(
            self,
            algo_name,  # type: str
            problem_name,  # type: str
            path,  # type: Union[str, Path]
    ):  # type: (...) -> None
        """Add a path to a performance history.

        Args:
            algo_name: The name of the algorithm associated with the history.
            problem_name: The name of the problem associated with the history.
            path: The path to the history.

        Raises:
            FileNotFoundError: If the path to the history does not exist.
        """
        try:
            absolute_path = Path(path).resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                "The path to the history does not exist: {}.".format(path)
            )
        if algo_name not in self.__dict:
            self.__dict[algo_name] = dict()
        if problem_name not in self.__dict[algo_name]:
            self.__dict[algo_name][problem_name] = list()
        self.__dict[algo_name][problem_name].append(str(absolute_path))
        # The paths are converted to strings to be JSON serializable.

    def to_file(
            self,
            path,  # type: Union[str, Path]
            indent=None,  # type: Optional[int]
    ):  # type: (...) -> None
        """Save the histories paths to a JSON file.

        Args:
            path: The path where to save the JSON file.
            indent: The indent level of the JSON serialization.
        """
        with Path(path).open("w") as file:
            json.dump(self.__dict, file, indent=indent)

    def from_file(
            self,
            path,  # type: Union[str, Path]
    ):  # type: (...) -> None
        """Load paths to performance histories from a JSON file.

        Args:
            path: The path to the JSON file.
        """
        #        if not Path(path).is_file():
        #            raise FileNotFoundError(
        #                "The path to the JSON file does not exist: {}.".format(path)
        #            )
        with Path(path).open("r") as file:
            histories = json.load(file)
        for algo_name, problems in histories.items():
            for problem_name, paths in problems.items():
                for path in paths:
                    self.add_path(algo_name, problem_name, path)
