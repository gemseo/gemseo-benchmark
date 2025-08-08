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

from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from jinja2 import Environment
from jinja2 import FileSystemLoader

from gemseo_benchmark import join_substrings
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.report._figures import Figures

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo_benchmark import ConfigurationPlotOptions
    from gemseo_benchmark.problems.optimization_problem_configuration import (
        OptimizationProblemConfiguration,
    )
    from gemseo_benchmark.problems.problems_group import ProblemsGroup
    from gemseo_benchmark.results.results import Results


class FileName(enum.Enum):
    """The name of a report file."""

    ALGORTIHM_CONFIGURATION_RESULTS = "algorithm_configuration_results.rst"
    ALGORITHMS = "algorithms.rst"
    ALGORITHMS_CONFIGURATIONS_GROUP = "algorithms_configurations_group.rst"
    INDEX = "index.rst"
    PROBLEM = "problem.rst"
    PROBLEMS_LIST = "problems_list.rst"
    PROBLEM_RESULTS = "problem_results.rst"
    RESULTS = "results.rst"
    SUB_RESULTS = "sub_results.rst"


class DirectoryName(enum.Enum):
    """The name of a report directory."""

    PROBLEMS = "problems"
    RESULTS = "results"
    IMAGES = "images"
    BUILD = "_build"


class Report:
    """A benchmarking report."""

    __FILE_DIRECTORY: Final[Path] = Path(__file__).parent
    __CONF_PATH: Final[Path] = __FILE_DIRECTORY / "conf.py"
    __NOT_AVAILABLE: Final[str] = "N/A"
    __TEMPLATES_DIR_PATH: Final[Path] = __FILE_DIRECTORY / "templates"

    def __init__(
        self,
        root_directory_path: str | Path,
        algos_configurations_groups: Iterable[AlgorithmsConfigurations],
        problems_groups: Iterable[ProblemsGroup],
        histories_paths: Results,
        custom_algos_descriptions: Mapping[str, str] | None = None,
        max_eval_number_per_group: dict[str, int] | None = None,
        plot_settings: Mapping[str, ConfigurationPlotOptions] = READ_ONLY_EMPTY_DICT,
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
            plot_settings: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.

        Raises:
            ValueError: If an algorithm has no associated histories.
        """  # noqa: D205, D212, D415
        self.__plot_settings = plot_settings
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
            msg = (
                f"Missing histories for algorithm{'s' if len(algos_diff) > 1 else ''} "
                f"{', '.join([f'{name!r}' for name in sorted(algos_diff)])}."
            )
            raise ValueError(msg)

        self.__max_eval_numbers = max_eval_number_per_group or {
            group.name: None for group in problems_groups
        }

    def generate(
        self,
        to_html: bool = True,
        to_pdf: bool = False,
        infeasibility_tolerance: float = 0.0,
        plot_all_histories: bool = False,
        use_log_scale: bool = False,
        plot_only_median: bool = False,
        use_time_log_scale: bool = False,
        use_abscissa_log_scale: bool = False,
    ) -> None:
        """Generate the benchmarking report.

        Args:
            to_html: Whether to generate the report in HTML format.
            to_pdf: Whether to generate the report in PDF format.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            plot_all_histories: Whether to plot all the performance histories.
            use_log_scale: Whether to use a logarithmic scale on the value axis.
            plot_only_median: Whether to plot only the median and no other centile.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
        """
        self.__create_root_directory()
        self.__create_algos_file()
        self.__create_problems_files()
        self.__create_results_files(
            infeasibility_tolerance,
            plot_all_histories,
            use_log_scale,
            plot_only_median,
            use_time_log_scale,
            use_abscissa_log_scale,
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
                    library = OptimizationLibraryFactory().create(algo_name)
                except ValueError:
                    # The algorithm is unavailable
                    algos_descriptions[algo_name] = self.__NOT_AVAILABLE
                else:
                    algos_descriptions[algo_name] = library.ALGORITHM_INFOS[
                        algo_name
                    ].description

        # Create the file
        self.__fill_template(
            self.__root_directory / FileName.ALGORITHMS.value,
            FileName.ALGORITHMS.value,
            algorithms=dict(sorted(algos_descriptions.items())),
        )

    def __create_problems_files(self) -> None:
        """Create the files describing the problem configurations."""
        problems_dir = self.__root_directory / DirectoryName.PROBLEMS.value
        problems_dir.mkdir()

        # Create a file for each problem
        problems_paths = []
        problems = [problem for group in self.__problems_groups for problem in group]
        problems = sorted(problems, key=lambda pb: pb.name.lower())
        for problem in problems:
            # Create the problem file
            file_path = self.__get_problem_path(problem)
            self.__fill_template(
                file_path,
                FileName.PROBLEM.value,
                name=problem.name,
                description=problem.description,
                optimum=self.__NOT_AVAILABLE
                if problem.optimum is None
                else f"{problem.optimum:.6g}",
                target_values=problem.target_values,
            )
            problems_paths.append(
                file_path.relative_to(self.__root_directory).as_posix()
            )

        # Create the list of problems
        self.__fill_template(
            file_path=self.__root_directory / FileName.PROBLEMS_LIST.value,
            template_name=FileName.PROBLEMS_LIST.value,
            problems_paths=problems_paths,
        )

    def __get_problem_path(self, problem: OptimizationProblemConfiguration) -> Path:
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
        plot_only_median: bool = False,
        use_time_log_scale: bool = False,
        use_abscissa_log_scale: bool = False,
    ) -> None:
        """Create the files corresponding to the benchmarking results.

        Args:
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            plot_all_histories: Whether to plot all the performance histories.
            use_log_scale: Whether to use a logarithmic scale on the value axis.
            plot_only_median: Whether to plot only the median and no other centile.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.
        """
        self.__fill_template(
            self.__root_directory / FileName.RESULTS.value,
            FileName.RESULTS.value,
            documents=[
                self.__create_algorithms_group_files(
                    group,
                    infeasibility_tolerance,
                    plot_all_histories,
                    use_log_scale,
                    plot_only_median,
                    use_time_log_scale,
                    use_abscissa_log_scale,
                )
                for group in self.__algorithms_configurations_groups
            ],
        )

    def __create_algorithms_group_files(
        self,
        algorithm_configurations: AlgorithmsConfigurations,
        infeasibility_tolerance: float,
        plot_all_histories: bool,
        use_log_scale: bool,
        plot_only_median: bool,
        use_time_log_scale: bool,
        use_abscissa_log_scale: bool,
    ) -> str:
        """Create the results files of a group of algorithm configurations.

        Args:
            algorithm_configurations: The algorithm configurations.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            plot_all_histories: Whether to plot all the performance histories.
            use_log_scale: Whether to use a logarithmic scale on the value axis.
            plot_only_median: Whether to plot only the median and no other centile.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.

        Returns:
            The path to the main file.
        """
        results_root = self.__root_directory / DirectoryName.RESULTS.value
        configurations_dirname = join_substrings(algorithm_configurations.name)
        configurations_dir = results_root / configurations_dirname
        configurations_dir.mkdir()
        paths = []
        for group in self.__problems_groups:
            # Get the configurations with results for all the problems of the group
            actual_configurations = AlgorithmsConfigurations(
                *[
                    configuration
                    for configuration in algorithm_configurations
                    if set(self.__histories_paths.get_problems(configuration.name))
                    >= {problem.name for problem in group}
                ],
                name=algorithm_configurations.name,
            )
            if not actual_configurations:
                # There is no configuration to display for the group
                continue

            problems_dirname = join_substrings(group.name)
            paths.append(
                self.__create_problems_group_files(
                    group,
                    actual_configurations,
                    configurations_dir,
                    self.__root_directory
                    / DirectoryName.IMAGES.value
                    / configurations_dirname
                    / problems_dirname,
                    infeasibility_tolerance,
                    plot_all_histories,
                    use_log_scale,
                    plot_only_median,
                    use_time_log_scale,
                    use_abscissa_log_scale,
                )
                .relative_to(results_root)
                .as_posix()
            )

        # Create the file of the group of algorithm configurations
        configurations_path = configurations_dir.with_suffix(".rst")
        self.__fill_template(
            configurations_path,
            FileName.ALGORITHMS_CONFIGURATIONS_GROUP.value,
            name=algorithm_configurations.name,
            documents=paths,
        )
        return configurations_path.relative_to(self.__root_directory).as_posix()

    def __create_problems_group_files(
        self,
        problems: ProblemsGroup,
        algorithm_configurations: AlgorithmsConfigurations,
        directory_path: Path,
        figures_dir: Path,
        infeasibility_tolerance: float,
        plot_all_histories: bool,
        use_log_scale: bool,
        plot_only_median: bool,
        use_time_log_scale: bool,
        use_abscissa_log_scale: bool,
    ) -> Path:
        """Create the results file of a group of algorithm configurations.

        Args:
            problems: The problems.
            algorithm_configurations: The algorithm configurations.
            directory_path: The path to the directory where to save the files.
            figures_dir: The path to the directory where to save the figures.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            plot_all_histories: Whether to plot all the performance histories.
            use_log_scale: Whether to use a logarithmic scale on the value axis.
            plot_only_median: Whether to plot only the median and no other centile.
            use_time_log_scale: Whether to use a logarithmic scale
                for the time axis.
            use_abscissa_log_scale: Whether to use a logarithmic scale
                for the abscissa axis.

        Returns:
            The path to the main file.
        """
        # Generate the figures
        figures_dir.mkdir(parents=True, exist_ok=False)
        plotter = Figures(
            algorithm_configurations,
            problems,
            self.__histories_paths,
            figures_dir,
            infeasibility_tolerance,
            self.__max_eval_numbers.get(problems.name),
            plot_settings=self.__plot_settings,
        )
        figures, tables = plotter.plot(
            plot_all_histories,
            use_log_scale,
            plot_only_median,
            use_time_log_scale,
            use_abscissa_log_scale,
        )

        # Create the file dedicated to the group of problems
        file_path = directory_path / f"{join_substrings(problems.name)}.rst"
        problems_dir = directory_path / join_substrings(problems.name)
        problems_dir.mkdir()
        self.__fill_template(
            file_path,
            FileName.SUB_RESULTS.value,
            algorithms_group_name=algorithm_configurations.name,
            algorithms_configurations_names=[
                algo_config.name for algo_config in algorithm_configurations
            ],
            problems_group_name=problems.name,
            problems_group_description=problems.description,
            data_profile=self.__get_relative_path(
                plotter.plot_data_profiles(use_abscissa_log_scale)
            ),
            problems_names=[problem.name for problem in problems],
            group_problems_paths=[
                self.__create_problem_results_files(
                    problem,
                    algorithm_configurations,
                    figures[problem.name],
                    problems_dir,
                    tables[problem.name],
                )
                .relative_to(directory_path)
                .as_posix()
                for problem in problems
            ],
        )
        return file_path

    def __create_problem_results_files(
        self,
        problem: OptimizationProblemConfiguration,
        algorithm_configurations: AlgorithmsConfigurations,
        figures: Figures.ProblemFigurePaths,
        directory_path: Path,
        tables: Figures.ProblemTablePaths,
    ) -> Path:
        """Create the files dedicated to the results obtained on a single problem.

        This methods creates
        * one file to present the results of all the algorithm configurations,
        * one file per algorithm configuration to present its own results.

        Args:
            problem: The problem.
            algorithm_configurations: The algorithm configurations.
            figures: The paths to the figures dedicated to the problem.
            directory_path: The path to the directory where to save the files.
            tables: The paths to the tables dedicated to the problem.

        Returns:
            The path to the main file.
        """
        # Create the files that present the results of each algorithm configuration.
        problem_path = directory_path / join_substrings(problem.name)
        problem_path.mkdir()
        algorithm_configurations_results = []
        for algorithm_configuration in algorithm_configurations:
            file_path = (
                problem_path / join_substrings(algorithm_configuration.name)
            ).with_suffix(".rst")
            self.__fill_template(
                file_path,
                FileName.ALGORTIHM_CONFIGURATION_RESULTS.value,
                algorithm_configuration=algorithm_configuration,
                problem=problem,
                figures={
                    name.value: self.__get_relative_path(
                        figures[algorithm_configuration.name][name]
                    )
                    for name in figures[algorithm_configuration.name]
                },
                tables={
                    name.value: self.__get_relative_path(
                        tables[algorithm_configuration.name][name]
                    )
                    for name in tables[algorithm_configuration.name]
                },
            )
            algorithm_configurations_results.append(
                file_path.relative_to(directory_path).as_posix()
            )

        # Create the file that presents the results of all the algorithm configurations.
        file_path = problem_path.with_suffix(".rst")
        self.__fill_template(
            file_path,
            FileName.PROBLEM_RESULTS.value,
            algorithm_configurations=algorithm_configurations,
            algorithm_configurations_results=algorithm_configurations_results,
            problem=problem,
            figures={
                name.value: self.__get_relative_path(figures[name])
                for name in Figures._FigureFileName
                if name in figures
            },
            tables={
                name.value: self.__get_relative_path(tables[name])
                for name in Figures._TableFileName
                if name in tables
            },
        )
        return file_path

    def __get_relative_path(self, file_path: Path) -> str:
        """Return a POSIX path relative to the root directory."""
        return file_path.relative_to(self.__root_directory).as_posix()

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
