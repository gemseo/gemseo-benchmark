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
"""Generation of a benchmarking report"""
from __future__ import annotations

import enum
import os
from shutil import copy
from subprocess import call
from typing import Any, Iterable, Mapping

from jinja2 import Environment, FileSystemLoader

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.utils.py23_compat import Path
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
            self, root_directory_path: str | Path,
            algos_specifications: Mapping[str, Mapping[str, Any]],
            problems_groups: Iterable[ProblemsGroup], histories_paths: Results,
            custom_algos_descriptions: Mapping[str, str] = None
    ) -> None:
        """
        Args:
            root_directory_path: The path to the root directory of the report.
            algos_specifications: The compared algorithms and their options.
            problems_groups: The groups of reference problems.
            histories_paths: The paths to the reference histories for each algorithm
                and reference problem.
            custom_algos_descriptions: The descriptions of the MINAMO algorithms.

        Raises:
            ValueError: If an algorithm has no associated histories,
                or has a missing history for a given reference problem.
        """
        self.__root_directory = Path(root_directory_path)
        self.__algos_specs = algos_specifications
        self.__problems_groups = problems_groups
        self.__histories_paths = histories_paths
        if custom_algos_descriptions is None:
            custom_algos_descriptions = dict()
        self.__custom_algos_descriptions = custom_algos_descriptions
        algos_diff = set(algos_specifications) - set(histories_paths.algorithms)
        if algos_diff:
            raise ValueError(
                f"Missing histories for algorithm{'s' if len(algos_diff) > 1 else ''} "
                f"{', '.join([f'{name!r}' for name in algos_diff])}."
            )
        for algo_name in algos_specifications:
            problems_diff = set(
                problem.name for group in problems_groups for problem in group
            ) - set(histories_paths.get_problems(algo_name))
            if problems_diff:
                raise ValueError(
                    f"Missing histories for algorithm {algo_name!r} "
                    f"on problem{'s' if len(problems_diff) > 1 else ''} "
                    f"{', '.join([f'{name!r}' for name in problems_diff])}."
                )

    def generate_report(self, to_html: bool = True, to_pdf: bool = False, ) -> None:
        """Generate the benchmarking report.

        Args:
            to_html: Whether to generate the report in HTML format.
            to_pdf: Whether to generate the report in PDF format.
        """
        self.__create_root_directory()
        self.__create_algos_file()
        self.__create_problems_files()
        self.__create_groups_files()
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
        for algo_name in self.__algos_specs:
            if algo_name not in algos_descriptions:
                try:
                    library = OptimizersFactory().create(algo_name)
                except ImportError:
                    # The algorithm is unavailable
                    algos_descriptions[algo_name] = "N/A"
                else:
                    algos_descriptions[algo_name] = library.lib_dict[algo_name][
                        library.DESCRIPTION
                    ]

        # Create the file
        file_path = self.__root_directory / FileName.ALGORITHMS.value
        self.__fill_template(
            file_path,
            FileName.ALGORITHMS.value,
            algorithms=algos_descriptions
        )

    def __create_problems_files(self) -> None:
        """Create the files describing the benchmarking problems."""
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

    def __create_groups_files(self) -> None:
        """Create the files corresponding to the problems groups."""
        groups_paths = list()
        for problems_group in self.__problems_groups:
            # Create the directory dedicated to the group
            group_dir = (
                    self.__root_directory / DirectoryName.IMAGES.value /
                    self.__format_name(problems_group.name)
            )
            group_dir.mkdir(exist_ok=False)

            # Generate the figures
            group_profile = self.__compute_group_data_profile(problems_group, group_dir)
            problems_figures = self.__plot_problems_figures(problems_group, group_dir)

            # Create the file
            group_path = (
                    self.__root_directory / DirectoryName.GROUPS.value /
                    f"{self.__format_name(problems_group.name)}.rst"
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

    @staticmethod
    def __format_name(name: str) -> str:
        """Format a name for the report source paths.

        Args:
            name: The name.

        Returns:
            The formatted name.
        """
        return name.replace(" ", "_")

    def __compute_group_data_profile(
            self, group: ProblemsGroup, destination_dir: Path
    ) -> str:
        """Compute the data profile for a group of benchmarking problems.

        Args:
            group: The group of benchmarking problems.
            destination_dir: The destination directory for the data profile.

        Returns:
            The path to the data profile, relative to the report root directory.
        """
        group_path = destination_dir / FileName.DATA_PROFILE.value
        group.compute_data_profile(
            self.__algos_specs, self.__histories_paths, show=False,
            plot_path=group_path,
        )
        return group_path.relative_to(self.__root_directory).as_posix()

    def __plot_problems_figures(
            self, group: ProblemsGroup, group_dir: Path
    ) -> dict[str, dict[str, str]]:
        """Plot the results figures for each problem of a group.

        Args:
            group: The group of benchmarking problems.
            group_dir: The path to the directory where to save the figures.

        Returns:
            The paths to the figures.
            The keys are the names of the problems and the values are dictionaries
            mapping "data profile" and "histories" to the path of the corresponding
            figures.
        """
        figures = dict()
        for problem in group:
            problem_dir = group_dir / self.__format_name(problem.name)
            problem_dir.mkdir()
            figures[problem.name] = {
                "data_profile": self.__plot_problem_data_profile(problem, problem_dir),
                "histories": self.__plot_problem_histories(problem, problem_dir)
            }

        # Sort the keys of the dictionary
        figures = {
            key: figures[key] for key in sorted(figures.keys(), key=str.lower)
        }
        return figures

    def __plot_problem_data_profile(
            self, problem: Problem, destination_dir: Path
    ) -> str:
        """Plot the data profile of a problem.

        Args:
            problem: The benchmarking problem.
            destination_dir: The destination directory for the figure.

        Returns:
            The path to the figure, relative to the root of the report.
        """
        path = destination_dir / FileName.DATA_PROFILE.value
        problem.compute_data_profile(
            self.__algos_specs, self.__histories_paths, file_path=path
        )
        return path.relative_to(self.__root_directory).as_posix()

    def __plot_problem_histories(self, problem: Problem, destination_dir: Path) -> str:
        """Plot the performance histories of a problem.

        Args:
            problem: The benchmarking problem.
            destination_dir: The destination directory for the figure.

        Returns:
            The path to the figure, relative to the root of the report.
        """
        path = destination_dir / FileName.HISTORIES.value
        problem.plot_histories(
            self.__algos_specs, self.__histories_paths, show=False, file_path=path
        )
        return path.relative_to(self.__root_directory).as_posix()
