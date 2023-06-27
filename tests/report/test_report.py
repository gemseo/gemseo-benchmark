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
"""Tests for the generation of a benchmarking report."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from gemseo_benchmark.report.report import Report


@pytest.fixture(scope="module")
def problems_groups(group) -> list[mock.Mock]:
    """The groups of problems."""
    return [group]


def test_init_missing_algorithms(
    tmp_path,
    unknown_algorithms_configurations,
    algorithm_configuration,
    unknown_algorithm_configuration,
    problems_groups,
    results,
):
    """Check the initialization of the report with missing algorithms histories."""
    results.algorithms = ["Another algo"]
    with pytest.raises(
        ValueError,
        match=f"Missing histories for algorithms "
        f"'{unknown_algorithm_configuration.name}', "
        f"'{algorithm_configuration.name}'.",
    ):
        Report(tmp_path, [unknown_algorithms_configurations], problems_groups, results)


@pytest.fixture(scope="function")
def report(tmp_path, algorithms_configurations, problems_groups, results) -> Report:
    """A benchmarking report."""
    return Report(tmp_path, [algorithms_configurations], problems_groups, results)


def test_generate_report_sources(tmp_path, report, algorithms_configurations, group):
    """Check the generation of the report sources."""
    report.generate(to_pdf=True)
    assert (tmp_path / "index.rst").is_file()
    assert (tmp_path / "algorithms.rst").is_file()
    assert (tmp_path / "results.rst").is_file()
    results_dir = tmp_path / "results"
    algorithms_configurations_name = algorithms_configurations.name.replace(" ", "_")
    assert (results_dir / f"{algorithms_configurations_name}.rst").is_file()
    assert (
        results_dir
        / algorithms_configurations_name
        / f"{group.name.replace(' ', '_')}.rst"
    ).is_file()
    assert (tmp_path / "_build" / "html" / "index.html").is_file()


@pytest.mark.skip(
    reason="The CI runner cannot execute the command `sphinx-build -M latexpdf`.",
)
def test_generate_pdf(tmp_path, report):
    """Check the generation of the report in PDF."""
    report.generate(to_pdf=True)
    assert (tmp_path / "_build" / "latex" / "benchmarking_report.pdf").is_file()


def test_retrieve_description(
    tmp_path, unknown_algorithms_configurations, problems_groups, results
):
    """Check the retrieval of a GEMSEO algorithm description."""
    report = Report(
        tmp_path, [unknown_algorithms_configurations], problems_groups, results
    )
    ref_contents = [
        "Algorithms\n",
        "==========\n",
        "\n",
        "The following algorithms are considered in this benchmarking report.\n",
        "\n",
        "Algorithm\n",
        "   N/A\n",
        "\n",
        "SLSQP\n",
        "   Sequential Least-Squares Quadratic Programming (SLSQP) implemented in the "
        "SciPy library\n",
    ]
    report.generate()
    with open(tmp_path / "algorithms.rst") as file:
        contents = file.readlines()

    assert contents == ref_contents


def test_problems_descriptions_files(tmp_path, report, problem_a, problem_b):
    """Check the generation of the files describing the problems."""
    report.generate(to_html=False)
    assert (tmp_path / "problems_list.rst").is_file()
    assert (tmp_path / "problems" / f"{problem_a.name}.rst").is_file()
    assert (tmp_path / "problems" / f"{problem_b.name}.rst").is_file()


def test_figures(
    tmp_path, report, algorithms_configurations, problems_groups, problem_a, problem_b
):
    """Check the generation of the figures."""
    report.generate(to_html=False)
    group_dir = (
        tmp_path
        / "images"
        / algorithms_configurations.name.replace(" ", "_")
        / problems_groups[0].name.replace(" ", "_")
    )
    assert (group_dir / "data_profile.png").is_file()
    problem_dir = group_dir / problem_a.name.replace(" ", "_")
    assert (problem_dir / "data_profile.png").is_file()
    assert (problem_dir / "histories.png").is_file()
    problem_dir = group_dir / problem_b.name.replace(" ", "_")
    assert (problem_dir / "data_profile.png").is_file()
    assert (problem_dir / "histories.png").is_file()


@pytest.fixture(scope="package")
def incomplete_problem():
    """An incomplete benchmarking problem."""
    problem = mock.Mock()
    problem.optimum = None
    return problem


@pytest.fixture(scope="package")
def incomplete_group(incomplete_problem):
    """A group with an incomplete benchmarking problem."""
    group = mock.MagicMock()
    group.__iter__.return_value = [incomplete_problem]
    return group


def test_problem_without_optimum(
    tmp_path, incomplete_group, algorithms_configurations, results
):
    """Check the handling of a benchmarking problem without an optimum."""
    groups = [incomplete_group]
    report = Report(tmp_path, [algorithms_configurations], groups, results)
    with pytest.raises(AttributeError, match="The optimum of the problem is not set."):
        report.generate()


@pytest.fixture(scope="function")
def incomplete_results(
    algorithm_configuration, unknown_algorithm_configuration, problem_a, problem_b
) -> mock.Mock:
    """The results of the benchmarking."""
    results = mock.Mock()
    results.algorithms = [
        algorithm_configuration.name,
        unknown_algorithm_configuration.name,
    ]
    results.get_problems = mock.Mock(return_value=[problem_a.name])
    paths = [Path(__file__).parent / "history.json"]
    results.get_paths = mock.Mock(return_value=paths)
    return results


def test_incomplete_results(
    tmp_path, algorithms_configurations, problems_groups, group, incomplete_results
):
    """Check the generation of a report with incomplete results."""
    Report(
        tmp_path, [algorithms_configurations], problems_groups, incomplete_results
    ).generate()
    algorithms_configurations_name = algorithms_configurations.name.replace(" ", "_")
    group_name = group.name.replace(" ", "_")
    assert not (
        tmp_path / "images" / algorithms_configurations_name / group_name
    ).is_dir()
    assert not (
        tmp_path
        / "results"
        / algorithms_configurations.name.replace(" ", "_")
        / f"{group_name}.rst"
    ).is_file()
