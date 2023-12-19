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
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Generation of a benchmarking report."""

from __future__ import annotations

import enum
import os
from pathlib import Path
from shutil import copy
from subprocess import call
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from gemseo.algos.opt.opt_factory import OptimizersFactory
from jinja2 import Environment
from jinja2 import FileSystemLoader

from gemseo_benchmark import join_substrings
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo_benchmark.problems.problem import Problem
    from gemseo_benchmark.problems.problems_group import ProblemsGroup
    from gemseo_benchmark.results.results import Results


class FileName(enum.Enum):
    """The name of a report file."""

    INDEX = "index.rst"
    PROBLEM = "problem.rst"
    PROBLEMS_LIST = "problems_list.rst"
    SUB_RESULTS = "sub_results.rst"
    RESULTS = "results.rst"
    ALGORITHMS = "algorithms.rst"
    ALGORITHMS_CONFIGURATIONS_GROUP = "algorithms_configurations_group.rst"
    DATA_PROFILE = "data_profile.png"
    HISTORIES = "histories.png"


class DirectoryName(enum.Enum):
    """The name of a report directory."""

    PROBLEMS = "problems"
    RESULTS = "results"
    IMAGES = "images"
    BUILD = "_build"


class Report:
    """A benchmarking report."""

    __FILE_DIRECTORY: Final[Path] = Path(__file__).parent
    __TEMPLATES_DIR_PATH: Final[Path] = __FILE_DIRECTORY / "templates"
    __CONF_PATH: Final[Path] = __FILE_DIRECTORY / "conf.py"

    def __init__(
        self,
        root_directory_path: str | Path,
        algos_configurations_groups: Iterable[AlgorithmsConfigurations],
        problems_groups: Iterable[ProblemsGroup],
        histories_paths: Results,
        custom_algos_descriptions: Mapping[str, str] | None = None,
        max_eval_number_per_group: dict[str, int] | None = None,
    ) -> None:
        """
        Args:
            root_directory_path: The path to the root directory of the report.
            algos_configurations_groups: The groups of algorithms configurations.
            problems_groups: The groups of reference problems.
            histories_paths: The paths to the reference histories for each algorithm
                and reference problem.
            custom_algos_descriptions: Custom descriptions of the algorithms,
                to be printed in the report instead of the default ones coded in GEMSEO.
            max_eval_number_per_group: The maximum evaluations numbers to be displayed
                on the graphs of each group.
                The keys are the groups names and the values are the maximum
                evaluations numbers for the graphs of the group.
                If ``None``, all the evaluations are displayed.
                If the key of a group is missing, all the evaluations are displayed
                for the group.

        Raises:
            ValueError: If an algorithm has no associated histories.
        """  # noqa: D205, D212, D415
        self.__root_directory = Path(root_directory_path)
        self.__algorithms_configurations_groups = algos_configurations_groups
        self.__problems_groups = problems_groups
        self.__histories_paths = histories_paths
        if custom_algos_descriptions is None:
            custom_algos_descriptions = {}

        self.__custom_algos_descriptions = custom_algos_descriptions
        algos_diff = set().union(*[
            group.names for group in algos_configurations_groups
        ]) - set(histories_paths.algorithms)
        if algos_diff:
            raise ValueError(
                f"Missing histories for algorithm{'s' if len(algos_diff) > 1 else ''} "
                f"{', '.join([f'{name!r}' for name in sorted(algos_diff)])}."
            )

        self.__max_eval_numbers = max_eval_number_per_group or {
            group.name: None for group in problems_groups
        }

    def generate(
        self,
        to_html: bool = True,
        to_pdf: bool = False,
        infeasibility_tolerance: float = 0.0,
        plot_all_histories: bool = True,
        use_log_scale: bool = False,
    ) -> None:
        """Generate the benchmarking report.

        Args:
            to_html: Whether to generate the report in HTML format.
            to_pdf: Whether to generate the report in PDF format.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            plot_all_histories: Whether to plot all the performance histories.
            use_log_scale: Whether to use a logarithmic scale on the value axis.
        """
        self.__create_root_directory()
        self.__create_algos_file()
        self.__create_problems_files()
        self.__create_results_files(
            infeasibility_tolerance, plot_all_histories, use_log_scale
        )
        self.__create_index()
        self.__build_report(to_html, to_pdf)

    def __create_root_directory(self) -> None:
        """Create the source directory and basic files."""
        self.__root_directory.mkdir(exist_ok=True)
        # Create the subdirectories
        (self.__root_directory / "_static").mkdir(exist_ok=True)
        for directory in [DirectoryName.RESULTS.value, DirectoryName.IMAGES.value]:
            (self.__root_directory / directory).mkdir(exist_ok=True)
        # Create the configuration file
        copy(str(self.__CONF_PATH), str(self.__root_directory / self.__CONF_PATH.name))

    def __create_algos_file(self) -> None:
        """Create the file describing the algorithms."""
        # Get the descriptions of the algorithms
        algos_descriptions = dict(self.__custom_algos_descriptions)
        for algo_name in set().union(*[
            algos_configs_group.algorithms
            for algos_configs_group in self.__algorithms_configurations_groups
        ]):
            if algo_name not in algos_descriptions:
                try:
                    library = OptimizersFactory().create(algo_name)
                except ImportError:
                    # The algorithm is unavailable
                    algos_descriptions[algo_name] = "N/A"
                else:
                    algos_descriptions[algo_name] = (
                        library.descriptions[algo_name].description
                    )

        # Create the file
        self.__fill_template(
            self.__root_directory / FileName.ALGORITHMS.value,
            FileName.ALGORITHMS.value,
            algorithms=dict(sorted(algos_descriptions.items())),
        )

    def __create_problems_files(self) -> None:
        """Create the files describing the benchmarking problems.

        Raises:
            AttributeError: If the optimum of the problem is not set.
        """
        problems_dir = self.__root_directory / DirectoryName.PROBLEMS.value
        problems_dir.mkdir()

        # Create a file for each problem
        problems_paths = []
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
                ],
            )
            problems_paths.append(path.relative_to(self.__root_directory).as_posix())

        # Create the list of problems
        self.__fill_template(
            file_path=self.__root_directory / FileName.PROBLEMS_LIST.value,
            template_name=FileName.PROBLEMS_LIST.value,
            problems_paths=problems_paths,
        )

    def __get_problem_path(self, problem: Problem) -> Path:
        """Return the path to a problem file.

        Args:
            problem: The problem.

        Returns:
            The path to the problem file.
        """
        return (
            self.__root_directory / DirectoryName.PROBLEMS.value / f"{problem.name}.rst"
        )

    def __create_results_files(
        self,
        infeasibility_tolerance: float = 0.0,
        plot_all_histories: bool = True,
        use_log_scale: bool = False,
    ) -> None:
        """Create the files corresponding to the benchmarking results.

        Args:
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            plot_all_histories: Whether to plot all the performance histories.
            use_log_scale: Whether to use a logarithmic scale on the value axis.
        """
        results_root = self.__root_directory / DirectoryName.RESULTS.value
        algos_configs_groups_paths = []
        for algorithms_configurations_group in self.__algorithms_configurations_groups:
            results_paths = []
            for problems_group in self.__problems_groups:
                # Get the algorithms with results for all the problems of the group
                algorithms_configurations = AlgorithmsConfigurations(*[
                    algo_config
                    for algo_config in algorithms_configurations_group
                    if set(self.__histories_paths.get_problems(algo_config.name))
                    >= {problem.name for problem in problems_group}
                ])
                if not algorithms_configurations:
                    # There is no algorithm to display for the group
                    continue

                # Create the directory dedicated to the results of the group of
                # algorithms configurations on the group of problems
                results_dir = (
                    self.__root_directory
                    / DirectoryName.IMAGES.value
                    / join_substrings(algorithms_configurations_group.name)
                    / join_substrings(problems_group.name)
                )
                results_dir.mkdir(parents=True, exist_ok=False)

                # Generate the figures
                data_profile = self.__compute_data_profile(
                    problems_group,
                    algorithms_configurations,
                    results_dir,
                    infeasibility_tolerance,
                )
                problems_figures = self.__plot_problems_figures(
                    problems_group,
                    algorithms_configurations,
                    results_dir,
                    infeasibility_tolerance,
                    plot_all_histories,
                    use_log_scale,
                )

                # Create the file
                results_path = (
                    results_root
                    / join_substrings(algorithms_configurations_group.name)
                    / f"{join_substrings(problems_group.name)}.rst"
                )
                results_path.parent.mkdir(exist_ok=True)
                results_paths.append(results_path.relative_to(results_root).as_posix())
                algorithms_configurations_names = [
                    algo_config.name
                    for algo_config in algorithms_configurations_group.configurations
                ]
                self.__fill_template(
                    results_path,
                    FileName.SUB_RESULTS.value,
                    algorithms_group_name=algorithms_configurations_group.name,
                    algorithms_configurations_names=algorithms_configurations_names,
                    problems_group_name=problems_group.name,
                    problems_group_description=problems_group.description,
                    data_profile=data_profile,
                    problems_figures=problems_figures,
                )

            # Create the file of the group of algorithms configurations
            algos_configs_group_path = (
                results_root
                / f"{join_substrings(algorithms_configurations_group.name)}.rst"
            )
            self.__fill_template(
                algos_configs_group_path,
                FileName.ALGORITHMS_CONFIGURATIONS_GROUP.value,
                name=algorithms_configurations_group.name,
                documents=results_paths,
            )
            algos_configs_groups_paths.append(
                algos_configs_group_path.relative_to(self.__root_directory).as_posix()
            )

        # Create the file listing the problems groups
        self.__fill_template(
            self.__root_directory / FileName.RESULTS.value,
            FileName.RESULTS.value,
            documents=algos_configs_groups_paths,
        )

    def __create_index(self) -> None:
        """Create the index file of the reST report."""
        # Create the table of contents tree
        toctree_contents = [
            FileName.ALGORITHMS.value,
            FileName.PROBLEMS_LIST.value,
            FileName.RESULTS.value,
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
        initial_dir = Path.cwd()
        os.chdir(str(self.__root_directory))
        builders = []
        if to_html:
            builders.append("html")
        if to_pdf:
            builders.append("latexpdf")
        try:
            for builder in builders:
                call(
                    f"sphinx-build -M {builder} {self.__root_directory} "
                    f"{DirectoryName.BUILD.value}".split()
                )
        finally:
            os.chdir(initial_dir)

    def __compute_data_profile(
        self,
        group: ProblemsGroup,
        algorithms_configurations: AlgorithmsConfigurations,
        destination_dir: Path,
        infeasibility_tolerance: float = 0.0,
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
            algorithms_configurations,
            self.__histories_paths,
            show=False,
            plot_path=group_path,
            infeasibility_tolerance=infeasibility_tolerance,
            max_eval_number=self.__max_eval_numbers.get(group.name),
        )
        return group_path.relative_to(self.__root_directory).as_posix()

    def __plot_problems_figures(
        self,
        group: ProblemsGroup,
        algorithms_configurations: AlgorithmsConfigurations,
        group_dir: Path,
        infeasibility_tolerance: float = 0.0,
        plot_all_histories: bool = True,
        use_log_scale: bool = False,
    ) -> dict[str, dict[str, str]]:
        """Plot the results figures for each problem of a group.

        Args:
            group: The group of benchmarking problems.
            algorithms_configurations: The algorithms configurations.
            group_dir: The path to the directory where to save the figures.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            plot_all_histories: Whether to plot all the performance histories.
            use_log_scale: Whether to use a logarithmic scale on the value axis.

        Returns:
            The paths to the figures.
            The keys are the names of the problems and the values are dictionaries
            mapping "data profile" and "histories" to the path of the corresponding
            figures.
        """
        max_eval_number = self.__max_eval_numbers.get(group.name)
        figures = {}
        for problem in group:
            problem_dir = group_dir / join_substrings(problem.name)
            problem_dir.mkdir()
            figures[problem.name] = {
                "data_profile": self.__plot_problem_data_profile(
                    problem,
                    algorithms_configurations,
                    problem_dir,
                    infeasibility_tolerance,
                    max_eval_number,
                ),
                "histories": self.__plot_problem_histories(
                    problem,
                    algorithms_configurations,
                    problem_dir,
                    infeasibility_tolerance,
                    max_eval_number,
                    plot_all_histories,
                    use_log_scale,
                ),
            }

        # Sort the keys of the dictionary
        return {key: figures[key] for key in sorted(figures.keys(), key=str.lower)}

    def __plot_problem_data_profile(
        self,
        problem: Problem,
        algorithms_configurations: AlgorithmsConfigurations,
        destination_dir: Path,
        infeasibility_tolerance: float = 0.0,
        max_eval_number: int | None = None,
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
            algorithms_configurations,
            self.__histories_paths,
            file_path=path,
            infeasibility_tolerance=infeasibility_tolerance,
            max_eval_number=max_eval_number,
        )
        return path.relative_to(self.__root_directory).as_posix()

    def __plot_problem_histories(
        self,
        problem: Problem,
        algorithms_configurations: AlgorithmsConfigurations,
        destination_dir: Path,
        infeasibility_tolerance: float = 0.0,
        max_eval_number: int | None = None,
        plot_all_histories: bool = True,
        use_log_scale: bool = False,
    ) -> str:
        """Plot the performance histories of a problem.

        Args:
            problem: The benchmarking problem.
            algorithms_configurations: The algorithms configurations.
            destination_dir: The destination directory for the figure.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            max_eval_number: The maximum evaluations number to be displayed on the
                graph.
            plot_all_histories: Whether to plot all the performance histories.
            use_log_scale: Whether to use a logarithmic scale on the value axis.

        Returns:
            The path to the figure, relative to the root of the report.
        """
        path = destination_dir / FileName.HISTORIES.value
        problem.plot_histories(
            algorithms_configurations,
            self.__histories_paths,
            show=False,
            file_path=path,
            plot_all_histories=plot_all_histories,
            infeasibility_tolerance=infeasibility_tolerance,
            max_eval_number=max_eval_number,
            use_log_scale=use_log_scale,
        )
        return path.relative_to(self.__root_directory).as_posix()
