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
"""A class to implement a benchmarking scenario (solving and reporting)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Final

from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.benchmarker.benchmarker import Benchmarker
from gemseo_benchmark.report.report import Report
from gemseo_benchmark.results.results import Results

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo_benchmark.problems.problems_group import ProblemsGroup

LOGGER = logging.getLogger(__name__)


class Scenario:
    """A benchmarking scenario, including running of solvers and reporting."""

    __DATABASES_DIRNAME: Final[str] = "databases"
    __HISTORIES_DIRNAME: Final[str] = "histories"
    __REPORT_DIRNAME: Final[str] = "report"
    __RESULTS_FILENAME: Final[str] = "results.json"

    def __init__(
        self,
        algorithms_configurations_groups: Iterable[AlgorithmsConfigurations],
        outputs_path: str | Path,
    ) -> None:
        """
        Args:
            algorithms_configurations_groups: The groups of algorithms configurations
                to be benchmarked.
            outputs_path: The path to the directory where to save the output files
                (histories and report).

        Raises:
            ValueError: If the path to outputs directory does not exist.
        """  # noqa: D205, D212, D415
        if not Path(outputs_path).is_dir():
            raise NotADirectoryError(
                f"The path to the outputs directory does not exist: {outputs_path}."
            )

        self._algorithms_configurations_groups = algorithms_configurations_groups
        self._outputs_path = Path(outputs_path).resolve()
        self._histories_path = self._get_dir_path(self.__HISTORIES_DIRNAME)
        self._results_path = self._outputs_path / self.__RESULTS_FILENAME

    def execute(
        self,
        problems_groups: Iterable[ProblemsGroup],
        overwrite_histories: bool = False,
        skip_solvers: bool = False,
        skip_report: bool = False,
        generate_html_report: bool = True,
        generate_pdf_report: bool = False,
        infeasibility_tolerance: float = 0.0,
        save_databases: bool = False,
        number_of_processes: int = 1,
        use_threading: bool = False,
        custom_algos_descriptions: Mapping[str, str] | None = None,
        max_eval_number_per_group: dict[str, int] | None = None,
        plot_all_histories: bool = True,
        use_log_scale: bool = False,
    ) -> Results:
        """Execute the benchmarking scenario.

        Args:
            problems_groups: The groups of benchmarking problems.
            overwrite_histories: Whether to overwrite the performance histories.
            skip_solvers: Whether to skip the running of solvers.
            skip_report: Whether to skip the generation of the report.
            generate_html_report: Whether to generate the report in HTML format.
            generate_pdf_report: Whether to generate the report in PDF format.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            save_databases: Whether to save the databases of the optimizations.
            number_of_processes: The maximum number of simultaneous threads or
                processes used to parallelize the execution.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution.
            custom_algos_descriptions: Custom descriptions of the algorithms,
                to be printed in the report instead of the default ones coded in GEMSEO.
            max_eval_number_per_group: The maximum evaluations numbers to be displayed
                on the graphs of each group.
                The keys are the groups names and the values are the maximum
                evaluations numbers for the graphs of the group.
                If ``None``, all the evaluations are displayed.
                If the key of a group is missing, all the evaluations are displayed
                for the group.
            plot_all_histories: Whether to plot all the performance histories.
            use_log_scale: Whether to use a logarithmic scale on the value axis.

        Returns:
            The performance histories.
        """
        if not skip_solvers:
            LOGGER.info("Run the solvers on the benchmarking problems")
            self._run_solvers(
                problems_groups,
                overwrite_histories,
                save_databases,
                number_of_processes,
                use_threading,
            )

        if not skip_report:
            LOGGER.info("Generate the benchmarking report")
            self.__generate_report(
                problems_groups,
                generate_html_report,
                generate_pdf_report,
                infeasibility_tolerance,
                custom_algos_descriptions,
                max_eval_number_per_group,
                plot_all_histories,
                use_log_scale,
            )

        return Results(self._results_path)

    def _run_solvers(
        self,
        problems_groups: Iterable[ProblemsGroup],
        overwrite_histories: bool,
        save_databases: bool,
        number_of_processes: int,
        use_threading: bool,
    ) -> None:
        """Run the solvers on the benchmarking problems.

        Args:
            problems_groups: The groups of benchmarking problems.
            overwrite_histories: Whether to overwrite the performance histories.
            save_databases: Whether to save the databases of the optimizations.
            number_of_processes: The maximum number of simultaneous threads or
                processes used to parallelize the execution.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution.
        """
        Benchmarker(
            self._histories_path,
            self._results_path,
            self._get_dir_path(self.__DATABASES_DIRNAME) if save_databases else None,
        ).execute(
            {problem for group in problems_groups for problem in group},
            AlgorithmsConfigurations(*[
                algo_config
                for algos_configs_group in self._algorithms_configurations_groups
                for algo_config in algos_configs_group
            ]),
            overwrite_histories,
            number_of_processes,
            use_threading,
        )

    def _get_dir_path(self, name: str, overwrite: bool = False) -> Path:
        """Return the path to a directory.

        Args:
            name: The name of the directory.
            overwrite: Whether to remove and remake the directory if it already exists.

        Returns:
            The path to the directory.
        """
        path = self._outputs_path / name
        if path.is_dir() and overwrite:
            shutil.rmtree(path)

        path.mkdir(exist_ok=not overwrite)
        return path

    def __generate_report(
        self,
        problems_groups: Iterable[ProblemsGroup],
        generate_to_html: bool,
        generate_to_pdf: bool,
        infeasibility_tolerance: float,
        custom_algos_descriptions: Mapping[str, str],
        max_eval_number_per_group: dict[str, int],
        plot_all_histories: bool = True,
        use_log_scale: bool = False,
    ) -> None:
        """Generate the benchmarking report.

        Args:
            problems_groups: The groups of benchmarking problems.
            generate_to_html: Whether to generate the report in HTML format.
            generate_to_pdf: Whether to generate the report in PDF format.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            custom_algos_descriptions: Custom descriptions of the algorithms,
                to be printed in the report instead of the default ones coded in GEMSEO.
            max_eval_number_per_group: The maximum evaluations numbers to be displayed
                on the graphs of each group.
                The keys are the groups names and the values are the maximum
                evaluations numbers for the graphs of the group.
                If ``None``, all the evaluations are displayed.
                If the key of a group is missing, all the evaluations are displayed
                for the group.
            plot_all_histories: Whether to plot all the performance histories.
            use_log_scale: Whether to use a logarithmic scale on the value axis.
        """
        report = Report(
            self.__get_report_path(),
            self._algorithms_configurations_groups,
            problems_groups,
            Results(self._results_path),
            custom_algos_descriptions,
            max_eval_number_per_group,
        )
        report.generate(
            generate_to_html,
            generate_to_pdf,
            infeasibility_tolerance,
            plot_all_histories,
            use_log_scale,
        )

    def __get_report_path(self) -> Path:
        """Return the path to the report root directory."""
        return self._get_dir_path(self.__REPORT_DIRNAME, overwrite=True)
