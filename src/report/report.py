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
from os import chdir
from shutil import copy
from subprocess import call
from typing import Any, Iterable, List, Mapping, Optional, Union

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.utils.py23_compat import Path
from jinja2 import Environment, FileSystemLoader

from problems.problems_group import ProblemsGroup


class Report(object):
    """A benchmarking report."""
    TEMPLATES_DIR = Path(__file__).parent / "templates"
    INDEX_FILENAME = "index.rst"
    ALGOS_FILENAME = "algorithms.rst"
    GROUPS_LIST_FILENAME = "problems_groups.rst"
    GROUP_FILENAME = "group.rst"

    BASIC_SOURCES_DIR = Path(__file__).parent / "basic_sources"
    CONF_PATH = BASIC_SOURCES_DIR / "conf.py"
    MAKE_PATH = BASIC_SOURCES_DIR / "make.bat"
    MAKEFILE_PATH = BASIC_SOURCES_DIR / "Makefile"

    GROUPS_DIR = "groups"
    IMAGES_DIR = "images"

    def __init__(
            self,
            root_directory,  # type: Union[str, Path]
            algos_specifications,  # type: Mapping[str, Mapping[str, Any]]
            problems_groups,  # type: Iterable[ProblemsGroup]
            histories_paths,  # type: Mapping[str, Mapping[str, List[Union[str, Path]]]]
            minamo_algos_descriptions=None,  # type: Optional[Mapping[str, str]]
    ):  # type: (...) -> None
        """
        Args:
            root_directory: The path to the root directory of the report.
            algos_specifications: The compared algorithms and their options.
            problems_groups: The groups of reference problems.
            histories_paths: The paths to the reference histories for each algorithm
                and reference problem.
            minamo_algos_descriptions: The descriptions of the MINAMO algorithms.

        Raises:
            ValueError: If an algorithm has no associated histories,
                or has a missing history for a given reference problem.

        """
        self.__root_directory = Path(root_directory)
        self.__algos_specs = algos_specifications
        self.__problems_groups = problems_groups
        self.__histories_paths = histories_paths
        self.__minamo_algos_descriptions = minamo_algos_descriptions
        for algo_name in algos_specifications:
            if algo_name not in histories_paths:
                raise ValueError(
                    "Missing histories for algorithm '{}'".format(algo_name)
                )
            some_histories = histories_paths[an_algo]
            for a_group in problems_groups:
                for problem in problems_group:
                    if problem.name not in some_histories:
                        raise ValueError(
                            "Missing histories for algorithm {!r} on problem {!r}"
                            .format(algo_name, problem.name)
                        )

    def generate_report_sources(
            self,
            html_report=True,  # type: bool
            pdf_report=False,  # type: bool
    ):  # type: (...) -> None
        """Generate the source files of the benchmarking report.

        Args:
            html_report: Whether to generate the report in HTML format.
            pdf_report: Whether to generate the report in PDF format.
        """
        self.__create_root_directory()
        self.__create_algos_file()
        self.__create_groups_files()
        self.__create_index()
        self.__build_report(html_report, pdf_report)

    def __create_root_directory(self):  # type: (...) -> None
        """Create the source directory and basic files."""
        root_directory = self.__root_directory
        root_directory.mkdir(exist_ok=True)
        # Create the subdirectories
        (root_directory / "_static").mkdir(exist_ok=True)
        for directory in [Report.GROUPS_DIR, Report.IMAGES_DIR]:
            (root_directory / directory).mkdir(exist_ok=True)
        # Create the basic source files
        for source_file in (
                Report.CONF_PATH, Report.MAKE_PATH, Report.MAKEFILE_PATH
        ):
            copy(str(source_file), str(root_directory / source_file.name))

    def __create_algos_file(self):  # type: (...)-> None
        """Create the file describing the algorithms."""
        # Get the descriptions of the algorithms
        algos_descriptions = dict()
        for algo_name in self.__algos_specs:
            try:
                library = OptimizersFactory().create(algo_name)
                algos_descriptions[algo_name] = library.lib_dict[algo_name][
                    library.DESCRIPTION
                ]
            except ImportError:
                # The algorithm is unavailable
                if self.__minamo_algos_descriptions is not None \
                        and algo_name in self.__minamo_algos_descriptions:
                    algos_descriptions[algo_name] = self.__minamo_algos_descriptions[
                        algo_name]
                else:
                    algos_descriptions[algo_name] = "<No description available.>"

        # Create the file
        file_path = self.__root_directory / Report.ALGOS_FILENAME
        Report.__fill_template(
            file_path,
            Report.ALGOS_FILENAME,
            algorithms=algos_descriptions
        )

    def __create_groups_files(self):  # type: (...) -> None
        """Create the files corresponding to the problems groups."""
        groups_paths = list()
        for problems_group in self.__problems_groups:
            # Create the directory dedicated to the group
            group_directory = (self.__root_directory / Report.IMAGES_DIR /
                               Report.__format_group_name(problems_group.name))
            group_directory.mkdir(exist_ok=True)

            # Generate the data profile
            data_profile_path = group_directory / "data_profile.png"
            problems_group.compute_data_profile(
                self.__algos_specs, self.__histories_paths, show=False,
                plot_path=str(data_profile_path),
            )
            data_profile = ".. image:: /{}".format(
                data_profile_path.relative_to(self.__root_directory).as_posix()
            )

            # Create the file
            group_path = (self.__root_directory / Report.GROUPS_DIR /
                            "{}.rst".format(Report.__format_group_name(problems_group.name)))
            groups_paths.append(
                group_path.relative_to(self.__root_directory).as_posix()
            )
            Report.__fill_template(
                group_path,
                Report.GROUP_FILENAME,
                name=problems_group.name,
                description=a_group.description,
                problems={problem.name: problem.__doc__ for a_problem in problems_group},
                data_profile=data_profile,
            )

        # Create the file listing the problems groups
        groups_list_path = self.__root_directory / Report.GROUPS_LIST_FILENAME
        Report.__fill_template(
            groups_list_path, Report.GROUPS_LIST_FILENAME,
            documents=groups_paths
        )

    def __create_index(self):  # type: (...) -> None
        """Create the index file of the reST report."""
        # Create the table of contents tree
        toctree_contents = [Report.ALGOS_FILENAME, Report.GROUPS_LIST_FILENAME]

        # Create the file
        index_path = self.__root_directory / Report.INDEX_FILENAME
        Report.__fill_template(
            index_path, Report.INDEX_FILENAME, documents=toctree_contents
        )

    @staticmethod
    def __fill_template(
            file_path,  # type: Path
            template_name,  # type: str
            **kwargs
    ):  # type: (...) -> None
        """Fill a file template.

        Args:
            file_path: The path to the file to be written.
            template_name: The name of the file template.

        Returns:
            The filled file template.
        """
        file_loader = FileSystemLoader(Report.TEMPLATES_DIR)
        environment = Environment(loader=file_loader)
        template = environment.get_template(template_name)
        file_contents = template.render(**kwargs)
        with file_path.open("w") as file:
            file.write(file_contents)

    def __build_report(
            self,
            html_report=True,  # type: bool
            pdf_report=False,  # type: bool
    ):  # type: (...) -> None
        """Build the benchmarking report.

        Args:
            html_report: Whether to generate the report in HTML format.
            pdf_report: Whether to generate the report in PDF format.
        """
        root_directory = self.__root_directory
        chdir(str(root_directory))
        if html_report:
            call("make html", shell=True)
        if pdf_report:
            call("make latexpdf", shell=True)

    @staticmethod
    def __format_group_name(
            name,  # type: str
    ):  # type: (...) -> str
        """Format a group name for the report source paths.

        Args:
            name: The group name.

        Returns:
            The formatted group name.
        """
        return name.replace(" ", "_")
