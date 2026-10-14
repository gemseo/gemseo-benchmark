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
from gemseo.algos.opt.scipy_local.scipy_local import ScipyOpt

from gemseo_benchmark.report.report import Report
from gemseo_benchmark.report.report import csv_to_md_table


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


# TODO: generate report sources once and for all


@pytest.fixture
def report(tmp_path, algorithms_configurations, problems_groups, results) -> Report:
    """A benchmarking report."""
    return Report(tmp_path, [algorithms_configurations], problems_groups, results)


def test_generate_report_sources(tmp_path, report, algorithms_configurations, group):
    """Check the generation of the report sources."""
    report.generate()
    docs_dir = tmp_path / "docs"
    assert (docs_dir / "index.md").is_file()
    assert (docs_dir / "algorithms.md").is_file()
    assert (docs_dir / "results.md").is_file()
    results_dir = docs_dir / "results"
    algorithms_configurations_name = algorithms_configurations.name.replace(" ", "_")
    assert (results_dir / f"{algorithms_configurations_name}.md").is_file()
    assert (
        results_dir
        / algorithms_configurations_name
        / f"{group.name.replace(' ', '_')}.md"
    ).is_file()
    assert (tmp_path / "_build" / "html" / "index.html").is_file()


@pytest.mark.skip(
    reason="The CI runner cannot install mkdocs-to-pdf system dependencies."
)
def test_generate_pdf(tmp_path, report):
    """Check the generation of the report in PDF."""
    report.generate(to_pdf=True)
    assert (tmp_path / "_build" / "html" / "benchmarking_report.pdf").is_file()


@pytest.mark.parametrize(("to_pdf", "expected"), [(False, False), (True, True)])
def test_generate_properdocs_yml(to_pdf, expected):
    """Check the generation of the properdocs.yml content."""
    yml = Report._Report__generate_properdocs_yml(to_pdf)
    assert yml.startswith("site_name: Benchmarking Report\n")
    assert ("  - to-pdf:" in yml) is expected
    assert ("      output_path: ../benchmarking_report.pdf" in yml) is expected
    assert yml.endswith("\n")


@pytest.mark.parametrize(
    "custom_algos_descriptions", [None, {"Algorithm": "Description"}]
)
def test_algorithm_descriptions(
    tmp_path,
    unknown_algorithms_configurations,
    problems_groups,
    results,
    custom_algos_descriptions,
):
    """Check the description of the algorithms."""
    report = Report(
        tmp_path,
        [unknown_algorithms_configurations],
        problems_groups,
        results,
        custom_algos_descriptions,
    )
    description = "Description" if custom_algos_descriptions else "N/A"
    slsqp_description = ScipyOpt.ALGORITHM_INFOS["SLSQP"].description
    ref_contents = [
        "# Algorithms\n",
        "\n",
        "The following algorithms are considered in this benchmarking report.\n",
        "\n",
        "## Algorithm\n",
        "\n",
        f"{description}\n",
        "\n",
        "## SLSQP\n",
        "\n",
        f"{slsqp_description}\n",
    ]
    report.generate()
    with open(tmp_path / "docs" / "algorithms.md") as file:
        contents = file.readlines()

    assert contents == ref_contents


def test_problems_descriptions_files(tmp_path, report, problem_a, problem_b):
    """Check the generation of the files describing the problems."""
    report.generate(to_html=False)
    docs_dir = tmp_path / "docs"
    assert (docs_dir / "problems_list.md").is_file()
    assert (docs_dir / "problems" / f"{problem_a.name}.md").is_file()
    assert (docs_dir / "problems" / f"{problem_b.name}.md").is_file()


def test_figures(
    tmp_path, report, algorithms_configurations, problems_groups, problem_a, problem_b
):
    """Check the generation of the figures."""
    report.generate(to_html=False)
    group_dir = (
        tmp_path
        / "docs"
        / "images"
        / algorithms_configurations.name.replace(" ", "_")
        / problems_groups[0].name.replace(" ", "_")
    )
    assert (group_dir / "data_profile.png").is_file()
    problem_dir = group_dir / problem_a.name.replace(" ", "_")
    assert (problem_dir / "data_profile.png").is_file()
    assert (problem_dir / "performance_measure.png").is_file()
    assert (problem_dir / "performance_measure_focus.png").is_file()
    problem_dir = group_dir / problem_b.name.replace(" ", "_")
    assert (problem_dir / "data_profile.png").is_file()
    assert (problem_dir / "performance_measure.png").is_file()
    assert (problem_dir / "performance_measure_focus.png").is_file()


@pytest.fixture(scope="package")
def incomplete_problem():
    """An incomplete problem configuration."""
    problem = mock.Mock()
    problem.optimum = None
    return problem


@pytest.fixture(scope="package")
def incomplete_group(incomplete_problem):
    """A group with an incomplete problem configuration."""
    group = mock.MagicMock()
    group.__iter__.return_value = [incomplete_problem]
    return group


@pytest.mark.parametrize(
    ("fixture_name", "optimum"), [("problem_a", "1"), ("problem_b", "N/A")]
)
def test_problem_files(tmp_path, report, fixture_name, optimum, request) -> None:
    """Check the problem files."""
    report.generate()
    problem = request.getfixturevalue(fixture_name)
    name = problem.name
    with (tmp_path / "docs" / "problems" / name).with_suffix(".md").open("r") as file:
        assert file.read() == (
            f"# {name}\n\n"
            f"## Description\n\n"
            f"{problem.description}\n\n"
            f"Optimal feasible objective value: {optimum}.\n\n"
            f"## Target values\n\n"
            f"- {problem.target_values[0].performance_measure} (feasible)\n"
        )


@pytest.fixture
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
        tmp_path / "docs" / "images" / algorithms_configurations_name / group_name
    ).is_dir()
    assert not (
        tmp_path
        / "docs"
        / "results"
        / algorithms_configurations.name.replace(" ", "_")
        / f"{group_name}.md"
    ).is_file()


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("", ""),
        ("a,b\n", "| a | b |\n| --- | --- |"),
        ("a,b\n1,2\n3,4\n", "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"),
    ],
)
def test_csv_to_md_table(tmp_path, contents, expected):
    """Check the rendering of a CSV file as a Markdown table."""
    csv_path = tmp_path / "table.csv"
    csv_path.write_text(contents)
    assert csv_to_md_table(csv_path) == expected


def test_build_failure_propagates(report, monkeypatch):
    """Check that a report build failure raises instead of passing silently."""
    from subprocess import CalledProcessError

    def fail(cmd, *args, **kwargs):
        raise CalledProcessError(1, cmd)

    monkeypatch.setattr("gemseo_benchmark.report.report.check_call", fail)
    with pytest.raises(CalledProcessError):
        report.generate()
