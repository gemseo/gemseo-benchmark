"""A class to implement a benchmarking scenario (solving and reporting)."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo_benchmark.algorithms.algorithms_configurations import \
    AlgorithmsConfigurations
from gemseo_benchmark.benchmarker import Benchmarker
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.report.report import Report
from gemseo_benchmark.results.results import Results

LOGGER = logging.getLogger(__name__)


class Scenario(object):
    """A benchmarking scenario, including running of solvers and reporting."""
    __DATABASES_DIRNAME = "databases"
    __HISTORIES_DIRNAME = "histories"
    __PSEVEN_LOGS_DIRNAME = "pseven_logs"
    __REPORT_DIRNAME = "report"
    __RESULTS_FILENAME = "results.json"

    def __init__(
        self, algorithms: AlgorithmsConfigurations, outputs_path: str | Path
    ) -> None:
        """
        Args:
            algorithms: The algorithms configurations to be benchmarked.
            outputs_path: The path to the directory where to save the output files
                (histories and report).

        Raises:
            ValueError: If the path to outputs directory does not exist.
        """
        if not Path(outputs_path).is_dir():
            raise NotADirectoryError(
                f"The path to the outputs directory does not exist: {outputs_path}."
            )

        self._algorithms = algorithms
        self._outputs_path = Path(outputs_path).resolve()

    def execute(
        self,
        problems_groups: Iterable[ProblemsGroup],
        overwrite_histories: bool = False,
        skip_solvers: bool = False,
        skip_report: bool = False,
        html_report: bool = True,
        pdf_report: bool = False,
        infeasibility_tolerance: float = 0.0,
        save_databases: bool = False,
        save_pseven_logs: bool = False
    ) -> None:
        """Execute the benchmarking scenario.

        Args:
            problems_groups: The groups of benchmarking problems.
            overwrite_histories: Whether to overwrite the performance histories.
            skip_solvers: Whether to skip the running of solvers.
            skip_report: Whether to skip the generation of the report.
            html_report: Whether to generate the report in HTML format.
            pdf_report: Whether to generate the report in PDF format.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
            save_databases: Whether to save the databases of the optimizations.
            save_pseven_logs: Whether to save the logs of pSeven.
        """
        if not skip_solvers:
            LOGGER.info("Run the solvers on the benchmarking problems")
            self._run_solvers(
                problems_groups, overwrite_histories, save_databases, save_pseven_logs
            )

        if not skip_report:
            LOGGER.info("Generate the benchmarking report")
            self.__generate_report(
                problems_groups, html_report, pdf_report, infeasibility_tolerance
            )

    def _run_solvers(
        self,
        problems_groups: Iterable[ProblemsGroup],
        overwrite_histories: bool,
        save_databases: bool,
        save_pseven_logs: bool
    ) -> None:
        """Run the solvers on the benchmarking problems.

        Args:
            problems_groups: The groups of benchmarking problems.
            overwrite_histories: Whether to overwrite the performance histories.
            save_databases: Whether to save the databases of the optimizations.
            save_pseven_logs: Whether to save the logs of pSeven.
        """
        # Avoid creating a useless directory for the pSeven logs
        if not save_pseven_logs or not OptimizersFactory().is_available("PSEVEN"):
            save_pseven_logs = False
        else:
            from gemseo.algos.opt.lib_pseven import PSevenOpt
            save_pseven_logs = any(
                [
                    algorithm.algorithm_name in PSevenOpt().descriptions
                    for algorithm in self._algorithms
                ]
            )

        benchmarker = Benchmarker(
            self._get_histories_path(),
            self._get_results_path(),
            self._get_databases_path() if save_databases else None,
            self._get_pseven_logs_path() if save_pseven_logs else None
        )
        benchmarker.execute(
            {problem for group in problems_groups for problem in group},
            self._algorithms,
            overwrite_histories,
        )

    def _get_histories_path(self) -> Path:
        """Return the path to the histories directory."""
        return self._get_dir_path(self.__HISTORIES_DIRNAME)

    def _get_results_path(self) -> Path:
        """Return the path to the results file."""
        return self._outputs_path / self.__RESULTS_FILENAME

    def _get_databases_path(self) -> Path:
        """Return the path to the databases directory."""
        return self._get_dir_path(self.__DATABASES_DIRNAME)

    def _get_pseven_logs_path(self) -> Path:
        """Return the path to the pSeven output files directory."""
        return self._get_dir_path(self.__PSEVEN_LOGS_DIRNAME)

    def _get_dir_path(self, name: str, overwrite: bool = False) -> Path:
        """Return the path to a directory.

        Args:
            name: the name of the directory.

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
        to_html: bool,
        to_pdf: bool,
        infeasibility_tolerance: float
    ) -> None:
        """Generate the benchmarking report.

        Args:
            problems_groups: The groups of benchmarking problems.
            to_html: Whether to generate the report in HTML format.
            to_pdf: Whether to generate the report in PDF format.
            infeasibility_tolerance: The tolerance on the infeasibility measure.
        """
        report = Report(
            self.__get_report_path(),
            self._algorithms,
            problems_groups,
            Results(self._get_results_path())
        )
        report.generate(to_html, to_pdf, infeasibility_tolerance)

    def __get_report_path(self) -> Path:
        """Return the path to the report root directory."""
        return self._get_dir_path(self.__REPORT_DIRNAME, overwrite=True)
