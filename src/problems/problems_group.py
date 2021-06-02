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
from typing import Any, Iterable, Iterator, List, Mapping, Optional

from data_profiles.data_profile import DataProfile
from data_profiles.performance_history import PerformanceHistory
from problems.problem import Problem


class ProblemsGroup(object):
    """A group of reference problems for benchmarking.

    N.B. reference problems should be grouped based on common characteristics such as
    functions smoothness and constraint set geometry.

    Attributes:
        name: The name of the group of problems.
        description: The description of the group of problems.
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
        self.name = name
        self.__problems = problems
        self.description = description

    def __iter__(self):  # type: (...) -> Iterator[Problem]
        """Iterate on the problems of the group."""
        return iter(problem for problem in self.__problems)

    def is_algorithm_suited(
            self,
            name,  # type: str
    ):  # type: (...) -> bool
        """Check whether an algorithm is suited to all the problems in the group.

        Args:
            name: The name of the algorithm.

        Returns:
            True if the algorithm is suited, False otherwise.
        """
        return all(problem.is_algorithm_suited(name) for problem in self.__problems)

    def compute_targets(
            self,
            targets_number,  # type: int
            ref_algos_specs,  # type: Mapping[str, Mapping[str, Any]]
            feasible=True,  # type: bool
    ):  # type: (...) -> None
        """Generate targets for all the problems based on given reference algorithms.

        Args:
            targets_number: The number of targets to generate.
            ref_algos_specs: The names and options of the reference algorithms.
            feasible: Whether to generate only feasible targets.
        """
        for problem in self.__problems:
            problem.compute_targets(targets_number, ref_algos_specs, feasible)

    def compute_data_profile(
            self,
            algos_specs,  # type: Mapping[str, Mapping[str, Any]]
            histories_paths,  # type: Mapping[str, Mapping[str, List[Path]]]
            show=True,  # type: bool
            plot_path=None  # type: Optional[str]
    ):  # type: (...) -> None
        """Generate the data profiles of given algorithms relative to the problems.

        Args:
            algos_specs: The algorithms and their options.
            histories_paths: The paths to the reference histories for each algorithm.
            show: if True, show the plot.
            plot_path: The path where to save the plot.
                By default the plot is not saved.
        """
        # Initialize the data profile
        target_values = {
            problem.name: problem.target_values for problem in self.__problems
        }
        data_profile = DataProfile(target_values)

        # Generate the performance histories
        for algo_name, algo_options in algos_specs.items():
            for problem in self.__problems:
                for history_path in histories_paths[algo_name][problem.name]:
                    history = PerformanceHistory.from_file(history_path)
                    obj_values = history.objective_values
                    measures = history.infeasibility_measures
                    data_profile.add_history(
                        problem.name, algo_name, obj_values, measures
                    )

        # Plot and/or save the data profile
        data_profile.plot(show=show, path=plot_path)
