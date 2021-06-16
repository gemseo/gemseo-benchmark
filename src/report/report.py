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
import os
from shutil import copy
from subprocess import call
from typing import Any, Iterable, List, Mapping, Optional, Union

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.utils.py23_compat import Path
from jinja2 import Environment, FileSystemLoader

from problems.problems_group import ProblemsGroup


class Report(object):
    """A benchmarking report."""
    TEMPLATES_DIR_PATH = Path(__file__).parent / "templates"
    INDEX_FILENAME = "index.rst"
    ALGOS_FILENAME = "algorithms.rst"
    GROUPS_LIST_FILENAME = "problems_groups.rst"
    GROUP_FILENAME = "group.rst"

    CONF_PATH = Path(__file__).parent / "conf.py"

    GROUPS_DIR_NAME = "groups"
    IMAGES_DIR_NAME = "images"
    BUILD_DIR_NAME = "_build"

    def __init__(
            self,
            root_directory_path,  # type: Union[str, Path]
            algos_specifications,  # type: Mapping[str, Mapping[str, Any]]
            problems_groups,  # type: Iterable[ProblemsGroup]
            histories_paths,  # type: Mapping[str, Mapping[str, List[Union[str, Path]]]]
            custom_algos_descriptions=None,  # type: Optional[Mapping[str, str]]
    ):  # type: (...) -> None
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
        algos_diff = set(algos_specifications) - set(histories_paths)
        if algos_diff:
            raise ValueError(
                "Missing histories for algorithm{} {}".format(
                    "s" if len(algos_diff) > 1 else "",
                    ", ".join(["{!r}".format(name) for name in algos_diff])
                )
            )
        for algo_name in algos_specifications:
            problems_diff = set(
                problem.name for group in problems_groups for problem in group
            ) - set(histories_paths[algo_name])
            if problems_diff:
                raise ValueError(
                    "Missing histories for algorithm {!r} on problem{} {}".format(
                        algo_name, "s" if len(problems_diff) > 1 else "",
                        ", ".join(["{!r}".format(name) for name in problems_diff])
                    )
                )

    def generate_report(
            self,
            to_html=True,  # type: bool
            to_pdf=False,  # type: bool
    ):  # type: (...) -> None
        """Generate the benchmarking report.

        Args:
            to_html: Whether to generate the report in HTML format.
            to_pdf: Whether to generate the report in PDF format.
        """
        self.__create_root_directory()
        self.__create_algos_file()
        self.__create_groups_files()
        self.__create_index()
        self.__build_report(to_html, to_pdf)

    def __create_root_directory(self):  # type: (...) -> None
        """Create the source directory and basic files."""
        self.__root_directory.mkdir(exist_ok=True)
        # Create the subdirectories
        (self.__root_directory / "_static").mkdir(exist_ok=True)
        for directory in [Report.GROUPS_DIR_NAME, Report.IMAGES_DIR_NAME]:
            (self.__root_directory / directory).mkdir(exist_ok=True)
        # Create the configuration file
        copy(str(Report.CONF_PATH), str(self.__root_directory / Report.CONF_PATH.name))

    def __create_algos_file(self):  # type: (...)-> None
        """Create the file describing the algorithms."""
        # Get the descriptions of the algorithms
        algos_descriptions = dict(self.__custom_algos_descriptions)
        for algo_name in self.__algos_specs:
            if algo_name not in algos_descriptions:
                try:
                    library = OptimizersFactory().create(algo_name)
                    algos_descriptions[algo_name] = library.lib_dict[algo_name][
                        library.DESCRIPTION
                    ]
                except ImportError:
                    # The algorithm is unavailable
                    algos_descriptions[algo_name] = "N/A"

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
            group_directory = (self.__root_directory / Report.IMAGES_DIR_NAME /
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
            group_path = (self.__root_directory / Report.GROUPS_DIR_NAME / "{}.rst"
                          .format(Report.__format_group_name(problems_group.name)))
            groups_paths.append(
                group_path.relative_to(self.__root_directory).as_posix()
            )
            Report.__fill_template(
                group_path,
                Report.GROUP_FILENAME,
                name=problems_group.name,
                description=problems_group.description,
                problems={problem.name: problem.__doc__ for problem in problems_group},
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
            **kwargs  # type: Any
    ):  # type: (...) -> None
        """Fill a file template.

        Args:
            file_path: The path to the file to be written.
            template_name: The name of the file template.

        Returns:
            The filled file template.
        """
        file_loader = FileSystemLoader(Report.TEMPLATES_DIR_PATH)
        environment = Environment(loader=file_loader)
        template = environment.get_template(template_name)
        file_contents = template.render(**kwargs)
        with file_path.open("w") as file:
            file.write(file_contents)

    def __build_report(
            self,
            to_html=True,  # type: bool
            to_pdf=False,  # type: bool
    ):  # type: (...) -> None
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
                call("sphinx-build -M {} {} {}".format(
                    builder, self.__root_directory, Report.BUILD_DIR_NAME
                ), shell=True)
        finally:
            os.chdir(initial_dir)

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
