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

import csv
import enum
from pathlib import Path
from subprocess import check_call
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


def csv_to_md_table(csv_path: Path) -> str:
    """Render a CSV file as a Markdown table.

    Args:
        csv_path: The path to the CSV file.
            The file must not contain a pipe character (`|`).

    Returns:
        The Markdown table, or an empty string if the CSV file is empty.
    """
    with csv_path.open() as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(x for x in [header, sep, body] if x)


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

    ALGORTIHM_CONFIGURATION_RESULTS = "algorithm_configuration_results.md"
    ALGORITHMS = "algorithms.md"
    ALGORITHMS_CONFIGURATIONS_GROUP = "algorithms_configurations_group.md"
    INDEX = "index.md"
    PROBLEM = "problem.md"
    PROBLEMS_LIST = "problems_list.md"
    PROBLEM_RESULTS = "problem_results.md"
    RESULTS = "results.md"
    SUB_RESULTS = "sub_results.md"


class DirectoryName(enum.Enum):
    """The name of a report directory."""

    PROBLEMS = "problems"
    RESULTS = "results"
    IMAGES = "images"
    BUILD = "_build"


class Report:
    """A benchmarking report."""

    __FILE_DIRECTORY: Final[Path] = Path(__file__).parent
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
                If `None`, all the evaluations are displayed.
                If the key of a group is missing, all the evaluations are displayed
                for the group.
            plot_settings: The keyword arguments of `matplotlib.axes.Axes.plot`
                for each algorithm configuration.

        Raises:
            ValueError: If an algorithm has no associated histories.
        """  # noqa: D205, D212, D415
        self.__plot_settings = plot_settings
        self.__root_directory = Path(root_directory_path)
        self.__docs_directory = self.__root_directory / "docs"
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
        self.__docs_directory.mkdir(exist_ok=True)
        for directory in [DirectoryName.RESULTS.value, DirectoryName.IMAGES.value]:
            (self.__docs_directory / directory).mkdir(exist_ok=True)

    def __create_algos_file(self) -> None:
        """Create the file describing the algorithms."""
        algos_descriptions = dict(self.__custom_algos_descriptions)
        for algo_name in set().union(*[
            algos_configs_group.algorithms
            for algos_configs_group in self.__algorithms_configurations_groups
        ]):
            if algo_name not in algos_descriptions:
                try:
                    library = OptimizationLibraryFactory().create(algo_name)
                except ValueError:
                    algos_descriptions[algo_name] = self.__NOT_AVAILABLE
                else:
                    algos_descriptions[algo_name] = library.ALGORITHM_INFOS[
                        algo_name
                    ].description

        self.__fill_template(
            self.__docs_directory / FileName.ALGORITHMS.value,
            FileName.ALGORITHMS.value,
            algorithms=dict(sorted(algos_descriptions.items())),
        )

    def __create_problems_files(self) -> None:
        """Create the files describing the problem configurations."""
        problems_dir = self.__docs_directory / DirectoryName.PROBLEMS.value
        problems_dir.mkdir()

        problems_paths = []
        problems = [problem for group in self.__problems_groups for problem in group]
        problems = sorted(problems, key=lambda pb: pb.name.lower())
        for problem in problems:
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
                file_path.relative_to(self.__docs_directory).as_posix()
            )

        self.__fill_template(
            file_path=self.__docs_directory / FileName.PROBLEMS_LIST.value,
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
            self.__docs_directory / DirectoryName.PROBLEMS.value / f"{problem.name}.md"
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
            self.__docs_directory / FileName.RESULTS.value,
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
        results_root = self.__docs_directory / DirectoryName.RESULTS.value
        configurations_dirname = join_substrings(algorithm_configurations.name)
        configurations_dir = results_root / configurations_dirname
        configurations_dir.mkdir()
        paths = []
        for group in self.__problems_groups:
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
                continue

            problems_dirname = join_substrings(group.name)
            paths.append(
                self
                .__create_problems_group_files(
                    group,
                    actual_configurations,
                    configurations_dir,
                    self.__docs_directory
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

        configurations_path = configurations_dir.with_suffix(".md")
        self.__fill_template(
            configurations_path,
            FileName.ALGORITHMS_CONFIGURATIONS_GROUP.value,
            name=algorithm_configurations.name,
            documents=paths,
        )
        return configurations_path.relative_to(self.__docs_directory).as_posix()

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

        file_path = directory_path / f"{join_substrings(problems.name)}.md"
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
                self
                .__create_problem_results_files(
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
        problem_path = directory_path / join_substrings(problem.name)
        problem_path.mkdir()
        algorithm_configurations_results = []
        for algorithm_configuration in algorithm_configurations:
            file_path = (
                problem_path / join_substrings(algorithm_configuration.name)
            ).with_suffix(".md")
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

        file_path = problem_path.with_suffix(".md")
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
        """Return a POSIX path relative to the docs directory."""
        return file_path.relative_to(self.__docs_directory).as_posix()

    def __create_index(self) -> None:
        """Create the index file of the Markdown report."""
        index_path = self.__docs_directory / FileName.INDEX.value
        self.__fill_template(index_path, FileName.INDEX.value)

    def __fill_template(
        self, file_path: Path, template_name: str, **kwargs: Any
    ) -> None:
        """Fill a file template.

        Args:
            file_path: The path to the file to be written.
            template_name: The name of the file template.
        """
        docs_dir = self.__docs_directory
        file_loader = FileSystemLoader(Report.__TEMPLATES_DIR_PATH)
        environment = Environment(
            loader=file_loader,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        environment.filters["stem"] = lambda p: Path(p).stem
        environment.filters["csv_to_md_table"] = lambda rel_path: csv_to_md_table(
            docs_dir / rel_path
        )
        template = environment.get_template(template_name)
        file_contents = template.render(**kwargs)
        with file_path.open("w") as file:
            file.write(file_contents)

    @staticmethod
    def __generate_properdocs_yml(to_pdf: bool) -> str:
        """Generate the content of the properdocs.yml configuration file.

        Args:
            to_pdf: Whether to include the to-pdf plugin for PDF generation.

        Returns:
            The YAML content as a string.
        """
        lines = [
            "site_name: Benchmarking Report",
            "docs_dir: docs",
            "site_dir: _build/html",
            "use_directory_urls: false",
            "theme:",
            "  name: material",
            "plugins:",
            "  - search",
        ]
        if to_pdf:
            lines += [
                "  - to-pdf:",
                "      output_path: ../benchmarking_report.pdf",
            ]
        return "\n".join(lines) + "\n"

    def __build_report(self, to_html: bool = True, to_pdf: bool = False) -> None:
        """Build the benchmarking report.

        Args:
            to_html: Whether to generate the report in HTML format.
            to_pdf: Whether to generate the report in PDF format.
        """
        if not to_html and not to_pdf:
            return
        properdocs_yml_path = self.__root_directory / "properdocs.yml"
        properdocs_yml_path.write_text(self.__generate_properdocs_yml(to_pdf))
        check_call(["properdocs", "build", "--config-file", str(properdocs_yml_path)])
