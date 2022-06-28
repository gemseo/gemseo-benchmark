# -*- coding: utf-8 -*-
# Copyright 2022 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
"""Generation of a benchmarking report"""
from __future__ import annotations

import enum
import os
from pathlib import Path
from shutil import copy
from subprocess import call
from typing import Any, Iterable, Mapping

from jinja2 import Environment, FileSystemLoader

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo_benchmark import join_substrings
from gemseo_benchmark.algorithms.algorithms_configurations import \
    AlgorithmsConfigurations
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.results.results import Results


class FileName(enum.Enum):
    """The name of a report file."""
    CONF = "conf.py"
    INDEX = "index.rst"
    PROBLEM = "problem.rst"
    PROBLEMS_LIST = "problems_list.rst"
    GROUP = "group.rst"
    GROUPS_LIST = "problems_groups.rst"
    ALGORITHMS = "algorithms.rst"
    DATA_PROFILE = "data_profile.png"
    HISTORIES = "histories.png"


class DirectoryName(enum.Enum):
    """The name of a report directory."""
    PROBLEMS = "problems"
    GROUPS = "groups"
    IMAGES = "images"
    BUILD = "_build"


class Report(object):
    """A benchmarking report."""
    __FILE_DIRECTORY = Path(__file__).parent
    __TEMPLATES_DIR_PATH = __FILE_DIRECTORY / "templates"
    __CONF_PATH = __FILE_DIRECTORY / "conf.py"

    def __init__(
            self,
            root_directory_path: str | Path,
            algos_configurations: AlgorithmsConfigurations,
            problems_groups: Iterable[ProblemsGroup],
            histories_paths: Results,
            custom_algos_descriptions: Mapping[str, str] = None,
            max_eval_number_per_group: dict[str, int] = None
    ) -> None:
        """
        Args:
            root_directory_path: The path to the root directory of the report.
            algos_configurations: The algorithms configurations.
            problems_groups: The groups of reference problems.
            histories_paths: The paths to the reference histories for each algorithm
                and reference problem.
            custom_algos_descriptions: The descriptions of the MINAMO algorithms.
            max_eval_number_per_group: The maximum evaluations numbers to be displayed
                on the graphs of each group.
                The keys are the groups names and the values are the maximum
                evaluations numbers for the graphs of the group.
                If ``None``, all the evaluations are displayed.
                If the key of a group is missing, all the evaluations are displayed
                for the group.

        Raises:
            ValueError: If an algorithm has no associated histories.
        """
        self.__root_directory = Path(root_directory_path)
        self.__algos_configs = algos_configurations
        self.__problems_groups = problems_groups
        self.__histories_paths = histories_paths
        if custom_algos_descriptions is None:
            custom_algos_descriptions = dict()

        self.__custom_algos_descriptions = custom_algos_descriptions
        algos_diff = set(algos_configurations.names) - set(histories_paths.algorithms)
        if algos_diff:
            raise ValueError(
                f"Missing histories for algorithm{'s' if len(algos_diff) > 1 else ''} "
                f"{', '.join([f'{name!r}' for name in sorted(algos_diff)])}."
            )

        self.__max_eval_numbers = max_eval_number_per_group or {
            group.name: None for group in problems_groups
        }

    def generate_report(
            self,
            to_html: bool = True,
            to_pdf: bool = False,
            infeasibility_tolerance: float = 0.0,
    ) -> None:
        """Generate the benchmarking report.

        Args:
            to_html: Whether to generate the report in HTML format.
            to_pdf: Whether to generate the report in PDF format.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
        """
        self.__create_root_directory()
        self.__create_algos_file()
        self.__create_problems_files()
        self.__create_groups_files(infeasibility_tolerance)
        self.__create_index()
        self.__build_report(to_html, to_pdf)

    def __create_root_directory(self) -> None:
        """Create the source directory and basic files."""
        self.__root_directory.mkdir(exist_ok=True)
        # Create the subdirectories
        (self.__root_directory / "_static").mkdir(exist_ok=True)
        for directory in [DirectoryName.GROUPS.value, DirectoryName.IMAGES.value]:
            (self.__root_directory / directory).mkdir(exist_ok=True)
        # Create the configuration file
        copy(str(self.__CONF_PATH), str(self.__root_directory / self.__CONF_PATH.name))

    def __create_algos_file(self) -> None:
        """Create the file describing the algorithms."""
        # Get the descriptions of the algorithms
        algos_descriptions = dict(self.__custom_algos_descriptions)
        for algo_name in self.__algos_configs.algorithms:
            if algo_name not in algos_descriptions:
                try:
                    library = OptimizersFactory().create(algo_name)
                except ImportError:
                    # The algorithm is unavailable
                    algos_descriptions[algo_name] = "N/A"
                else:
                    algos_descriptions[algo_name] = library.descriptions[
                        algo_name
                    ].description

        # Create the file
        file_path = self.__root_directory / FileName.ALGORITHMS.value
        self.__fill_template(
            file_path,
            FileName.ALGORITHMS.value,
            algorithms=algos_descriptions
        )

    def __create_problems_files(self) -> None:
        """Create the files describing the benchmarking problems.

        Raises:
            AttributeError: If the optimum of the problem is not set.
        """
        problems_dir = self.__root_directory / DirectoryName.PROBLEMS.value
        problems_dir.mkdir()

        # Create a file for each problem
        problems_paths = list()
        problems = [problem for group in self.__problems_groups for problem in group]
        problems = sorted(problems, key=lambda pb: pb.name.lower())
        for problem in problems:
            # Create the problem file
            path = self.__get_problem_path(problem)
            # Skip duplicate problems
            if path.is_file():
                continue

            if problem.optimum is None:
                raise AttributeError("The optimum of the problem is not set.")

            self.__fill_template(
                path,
                FileName.PROBLEM.value,
                name=problem.name,
                description=problem.description,
                optimum=f"{problem.optimum:.6g}",
                target_values=[
                    f"{target.objective_value:.6g}" for target in problem.target_values
                ]
            )
            problems_paths.append(path.relative_to(self.__root_directory).as_posix())

        # Create the list of problems
        self.__fill_template(
            file_path=self.__root_directory / FileName.PROBLEMS_LIST.value,
            template_name=FileName.PROBLEMS_LIST.value,
            problems_paths=problems_paths
        )

    def __get_problem_path(self, problem: Problem) -> Path:
        """Return the path to a problem file.

        Args:
            problem: The problem.

        Returns:
            The path to the problem file.
        """
        return (
                self.__root_directory / DirectoryName.PROBLEMS.value /
                f"{problem.name}.rst"
        )

    def __create_groups_files(self, infeasibility_tolerance: float = 0.0) -> None:
        """Create the files corresponding to the problems groups.

        Args:
            infeasibility_tolerance: The tolerance on the infeasibility measure.
        """
        groups_paths = list()
        for problems_group in self.__problems_groups:

            # Get the algorithms with results for all the problems of the group
            problems_names = {problem.name for problem in problems_group}
            algorithms_configurations = AlgorithmsConfigurations(
                *[
                    algo_config for algo_config in self.__algos_configs if set(
                        self.__histories_paths.get_problems(algo_config.name)
                    ) >= problems_names
                ]
            )
            if not algorithms_configurations:
                # There is no algorithm to display for the group
                continue

            # Create the directory dedicated to the group
            group_dir = (
                    self.__root_directory / DirectoryName.IMAGES.value /
                    join_substrings(problems_group.name)
            )
            group_dir.mkdir(exist_ok=False)

            # Generate the figures
            group_profile = self.__compute_group_data_profile(
                problems_group, algorithms_configurations, group_dir,
                infeasibility_tolerance
            )
            problems_figures = self.__plot_problems_figures(
                problems_group, algorithms_configurations, group_dir,
                infeasibility_tolerance
            )

            # Create the file
            group_path = (
                    self.__root_directory / DirectoryName.GROUPS.value /
                    f"{join_substrings(problems_group.name)}.rst"
            )
            groups_paths.append(
                group_path.relative_to(self.__root_directory).as_posix()
            )
            self.__fill_template(
                group_path,
                FileName.GROUP.value,
                name=problems_group.name,
                description=problems_group.description,
                problems_figures=problems_figures,
                data_profile=group_profile,
            )

        # Create the file listing the problems groups
        groups_list_path = self.__root_directory / FileName.GROUPS_LIST.value
        self.__fill_template(
            groups_list_path, FileName.GROUPS_LIST.value,
            documents=groups_paths
        )

    def __create_index(self) -> None:
        """Create the index file of the reST report."""
        # Create the table of contents tree
        toctree_contents = [
            FileName.ALGORITHMS.value,
            FileName.PROBLEMS_LIST.value,
            FileName.GROUPS_LIST.value
        ]

        # Create the file
        index_path = self.__root_directory / FileName.INDEX.value
        self.__fill_template(
            index_path, FileName.INDEX.value, documents=toctree_contents
        )

    @staticmethod
    def __fill_template(file_path: Path, template_name: str, **kwargs: Any) -> None:
        """Fill a file template.

        Args:
            file_path: The path to the file to be written.
            template_name: The name of the file template.

        Returns:
            The filled file template.
        """
        file_loader = FileSystemLoader(Report.__TEMPLATES_DIR_PATH)
        environment = Environment(loader=file_loader)
        template = environment.get_template(template_name)
        file_contents = template.render(**kwargs)
        with file_path.open("w") as file:
            file.write(file_contents)

    def __build_report(self, to_html: bool = True, to_pdf: bool = False) -> None:
        """Build the benchmarking report.

        Args:
            to_html: Whether to generate the report in HTML format.
            to_pdf: Whether to generate the report in PDF format.
        """
        initial_dir = os.getcwd()
        os.chdir(str(self.__root_directory))
        builders = list()
        if to_html:
            builders.append("html")
        if to_pdf:
            builders.append("latexpdf")
        try:
            for builder in builders:
                call(
                    f"sphinx-build -M {builder} {self.__root_directory} "
                    f"{DirectoryName.BUILD.value}",
                    shell=True
                )
        finally:
            os.chdir(initial_dir)

    def __compute_group_data_profile(
            self,
            group: ProblemsGroup,
            algorithms_configurations: AlgorithmsConfigurations,
            destination_dir: Path,
            infeasibility_tolerance: float = 0.0
    ) -> str:
        """Compute the data profile for a group of benchmarking problems.

        Args:
            group: The group of benchmarking problems.
            algorithms_configurations: The algorithms configurations.
            destination_dir: The destination directory for the data profile.
            infeasibility_tolerance: The tolerance on the infeasibility measure.

        Returns:
            The path to the data profile, relative to the report root directory.
        """
        group_path = destination_dir / FileName.DATA_PROFILE.value
        group.compute_data_profile(
            algorithms_configurations, self.__histories_paths, show=False,
            plot_path=group_path, infeasibility_tolerance=infeasibility_tolerance,
            max_eval_number=self.__max_eval_numbers.get(group.name)
        )
        return group_path.relative_to(self.__root_directory).as_posix()

    def __plot_problems_figures(
            self,
            group: ProblemsGroup,
            algorithms_configurations: AlgorithmsConfigurations,
            group_dir: Path,
            infeasibility_tolerance: float = 0.0
    ) -> dict[str, dict[str, str]]:
        """Plot the results figures for each problem of a group.

        Args:
            group: The group of benchmarking problems.
            algorithms_configurations: The algorithms configurations.
            group_dir: The path to the directory where to save the figures.
            infeasibility_tolerance: The tolerance on the infeasibility measure.

        Returns:
            The paths to the figures.
            The keys are the names of the problems and the values are dictionaries
            mapping "data profile" and "histories" to the path of the corresponding
            figures.
        """
        max_eval_number = self.__max_eval_numbers.get(group.name)
        figures = dict()
        for problem in group:
            problem_dir = group_dir / join_substrings(problem.name)
            problem_dir.mkdir()
            figures[problem.name] = {
                "data_profile": self.__plot_problem_data_profile(
                    problem, algorithms_configurations, problem_dir,
                    infeasibility_tolerance, max_eval_number
                ),
                "histories": self.__plot_problem_histories(
                    problem, algorithms_configurations, problem_dir,
                    infeasibility_tolerance, max_eval_number
                )
            }

        # Sort the keys of the dictionary
        figures = {
            key: figures[key] for key in sorted(figures.keys(), key=str.lower)
        }
        return figures

    def __plot_problem_data_profile(
            self,
            problem: Problem,
            algorithms_configurations: AlgorithmsConfigurations,
            destination_dir: Path,
            infeasibility_tolerance: float = 0.0,
            max_eval_number: int = None
    ) -> str:
        """Plot the data profile of a problem.

        Args:
            problem: The benchmarking problem.
            algorithms_configurations: The algorithms configurations.
            destination_dir: The destination directory for the figure.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum evaluations number to be displayed on the
                graph.

        Returns:
            The path to the figure, relative to the root of the report.
        """
        path = destination_dir / FileName.DATA_PROFILE.value
        problem.compute_data_profile(
            algorithms_configurations, self.__histories_paths, file_path=path,
            infeasibility_tolerance=infeasibility_tolerance,
            max_eval_number=max_eval_number
        )
        return path.relative_to(self.__root_directory).as_posix()

    def __plot_problem_histories(
            self,
            problem: Problem,
            algorithms_configurations: AlgorithmsConfigurations,
            destination_dir: Path,
            infeasibility_tolerance: float = 0.0,
            max_eval_number: int = None
    ) -> str:
        """Plot the performance histories of a problem.

        Args:
            problem: The benchmarking problem.
            algorithms_configurations: The algorithms configurations.
            destination_dir: The destination directory for the figure.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum evaluations number to be displayed on the
                graph.

        Returns:
            The path to the figure, relative to the root of the report.
        """
        path = destination_dir / FileName.HISTORIES.value
        problem.plot_histories(
            algorithms_configurations, self.__histories_paths, show=False,
            file_path=path, plot_all_histories=True,
            infeasibility_tolerance=infeasibility_tolerance,
            max_eval_number=max_eval_number
        )
        return path.relative_to(self.__root_directory).as_posix()
