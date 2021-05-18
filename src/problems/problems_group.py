# -*- coding: utf-8 -*-
# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Grouping of reference problems for benchmarking."""
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from data_profiles.data_profile import DataProfile
from data_profiles.performance_history import PerformanceHistory
from problems.problem import Problem


class ProblemsGroup(object):
    """A group of reference problems for benchmarking.

    N.B. reference problems should be grouped based on common characteristics such as
    functions smoothness and constraint set geometry.
    """

    def __init__(
            self,
            name,  # type: str
            problems,  # type: Iterable[Problem]
            description=None,  # type: Optional[str]
    ):
        """
        Args:
            name: The name of the group of problems.
            problems: The benchmarking problems of the group.
            description: The description of the group of problems.
        """
        self._name = name
        self._problems = problems
        self._description = description

    @property
    def name(self):  # type: (...) -> str
        """The name of the group of problems."""
        return self._name

    @property
    def description(self):  # type: (...) -> str
        """The description of the group of problems."""
        return self._description

    def __iter__(self):  # type: (...) -> Iterator[Problem]
        """Iterate on the problems of the group."""
        return iter(a_problem for a_problem in self._problems)

    def is_algorithm_suited(
            self,
            name,  # type: str
    ):  # type: (...) -> bool
        """Check whether an algorithm is suited to all the problems in the group.

        Args:
            name: The name of the algorithm.
        """
        return all(a_problem.is_algorithm_suited(name) for a_problem in self._problems)

    def generate_targets(
            self,
            targets_number,  # type: int
            reference_algorithms,  # type: Dict[str, Dict]
            feasible=True,  # type: bool
    ):  # type: (...) -> None
        """Generate targets for all the problems based on given reference algorithms.

        Args:
            targets_number: The number of targets to generate.
            reference_algorithms: The names and options of the reference algorithms.
            feasible: Whether to generate only feasible targets.
        """
        for a_problem in self._problems:
            a_problem.generate_targets(targets_number, reference_algorithms, feasible)

    def generate_data_profile(
            self,
            algorithms,  # type: Dict[str, Dict]
            histories_paths,  # type: Dict[str, Dict[str, List[Path]]]
            show=True,  # type: bool
            destination_path=None  # type: Optional[str]
    ):  # type: (...) -> None
        """Generate the data profiles of given algorithms relative to the problems.

        Args:
            algorithms: The algorithms and their options.
            histories_paths: The paths to the reference histories for each algorithm.
            show: Whether to show the plot.
            destination_path: The path where to save the plot.
                By default the plot is not saved.
        """
        # Initialize the data profile
        target_values = {
            a_problem.name: a_problem.target_values for a_problem in self._problems
        }
        data_profile = DataProfile(target_values)

        # Generate the performance histories
        for an_algo_name, an_algo_options in algorithms.items():
            for a_problem in self._problems:
                for a_history_path in histories_paths[an_algo_name][a_problem.name]:
                    history = PerformanceHistory.load_from_file(a_history_path)
                    obj_values = history.objective_values
                    measures = history.infeasibility_measures
                    data_profile.add_history(
                        a_problem.name, an_algo_name, obj_values, measures
                    )

        # Plot and/or save the data profile
        data_profile.plot(show=show, destination_path=destination_path)
