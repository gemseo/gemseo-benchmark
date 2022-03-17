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
"""Tests for the generation of a benchmarking report"""
import shutil
from typing import Dict, List

import pytest

from gemseo.utils.py23_compat import mock, Path
from gemseo_benchmark.report.report import Report

ALGO_NAME = "SLSQP"


@pytest.fixture(scope="module")
def algos_specifications():  # type: (...) -> Dict[str, Dict]
    """The specifications of the algorithms."""
    return {ALGO_NAME: dict()}


def side_effect(algos_specifications, results, show=False, file_path=None):
    shutil.copyfile(str(Path(__file__).parent / "data_profile.png"), str(file_path))


@pytest.fixture(scope="module")
def problem_a():  # type: (...) -> mock.Mock
    """A problem."""
    problem = mock.Mock()
    problem.name = "Problem A"
    problem.description = "The description of problem A."
    problem.optimum = 1.0
    target = mock.Mock()
    target.objective_value = problem.optimum
    problem.target_values = [target]
    problem.compute_data_profile = mock.Mock(side_effect=side_effect)
    problem.plot_histories = mock.Mock(side_effect=side_effect)
    return problem


@pytest.fixture(scope="module")
def problem_b() -> mock.Mock:
    """Another problem."""
    problem = mock.Mock()
    problem.name = "Problem B"
    problem.description = "The description of problem B."
    problem.optimum = 2.0
    target = mock.Mock()
    target.objective_value = problem.optimum
    problem.target_values = [target]
    problem.compute_data_profile = mock.Mock(side_effect=side_effect)
    problem.plot_histories = mock.Mock(side_effect=side_effect)
    return problem


@pytest.fixture(scope="module")
def group(problem_a, problem_b):  # type: (...) -> mock.Mock
    """The group of problems."""
    group = mock.MagicMock()
    group.name = "A group"
    group.description = "The description of the group."
    group.__iter__.return_value = [problem_a, problem_b]

    def side_effect(algos_specifications, histories_paths, show, plot_path):
        shutil.copyfile(str(Path(__file__).parent / "data_profile.png"), str(plot_path))

    group.compute_data_profile = mock.Mock(side_effect=side_effect)
    return group


@pytest.fixture(scope="module")
def problems_groups(group):  # type: (...) -> List[mock.Mock]
    """The groups of problems."""
    return [group]


@pytest.fixture
def results(problem_a, problem_b):  # type: (...) -> mock.Mock
    """The results of the benchmarking."""
    results = mock.Mock()
    results.algorithms = [ALGO_NAME]
    results.get_problems = mock.Mock(return_value=[problem_a.name, problem_b.name])
    return results


def test_init_missing_algorithms(
        tmp_path, algos_specifications, problems_groups, results
):
    """Check the initialization of the report with missing algorithms histories."""
    results.algorithms = ["Another algo"]
    with pytest.raises(
            ValueError, match=f"Missing histories for algorithm '{ALGO_NAME}'."
    ):
        Report(tmp_path, algos_specifications, problems_groups, results)


def test_init_missing_problems(
        tmp_path, algos_specifications, problem_a, problem_b, problems_groups, results
):
    """Check the initialization of the report with missing problems histories."""
    results.get_problems = mock.Mock(return_value=["Another problem"])
    with pytest.raises(
            ValueError,
            match=f"Missing histories for algorithm '{ALGO_NAME}' on problems "
                  f"('{problem_a.name}', '{problem_b.name}')|('{problem_b.name}', "
                  f"'{problem_a.name}')."
    ):
        Report(tmp_path, algos_specifications, problems_groups, results)


def test_generate_report_sources(
        tmp_path, algos_specifications, problems_groups, results
):
    """Check the generation of the report sources."""
    report = Report(tmp_path, algos_specifications, problems_groups, results)
    report.generate_report(to_pdf=True)
    assert (tmp_path / "index.rst").is_file()
    assert (tmp_path / "algorithms.rst").is_file()
    assert (tmp_path / "problems_groups.rst").is_file()
    assert (tmp_path / "groups" / "A_group.rst").is_file()
    assert (tmp_path / "_build" / "html" / "index.html").is_file()
    assert (tmp_path / "_build" / "latex" / "benchmarking_report.pdf").is_file()


def test_retrieve_description(tmp_path, algos_specifications, problems_groups, results):
    """Check the retrieval of a GEMSEO algorithm description."""
    report = Report(tmp_path, algos_specifications, problems_groups, results)
    ref_contents = [
        "Algorithms\n",
        "==========\n",
        "\n",
        "The following algorithms are considered in this benchmarking report.\n",
        "\n",
        "SLSQP\n",
        "   Sequential Least-Squares Quadratic Programming (SLSQP) implemented in the "
        "SciPy library\n"
    ]
    report.generate_report()
    with open(tmp_path / "algorithms.rst") as file:
        contents = file.readlines()
    assert contents == ref_contents


def test_problems_descriptions_files(
        tmp_path, algos_specifications, problem_a, problem_b, problems_groups, results
):
    """Check the generation of the files describing the problems."""
    report = Report(tmp_path, algos_specifications, problems_groups, results)
    report.generate_report(to_html=False)
    assert (tmp_path / "problems_list.rst").is_file()
    assert (tmp_path / "problems" / f"{problem_a.name}.rst").is_file()
    assert (tmp_path / "problems" / f"{problem_b.name}.rst").is_file()


def test_figures(
        tmp_path, algos_specifications, problem_a, problem_b, problems_groups, results
):
    """Check the generation of the figures."""
    report = Report(tmp_path, algos_specifications, problems_groups, results)
    report.generate_report(to_html=False)
    group_dir = tmp_path / "images" / problems_groups[0].name.replace(" ", "_")
    assert (group_dir / "data_profile.png").is_file()
    problem_dir = group_dir / problem_a.name.replace(" ", "_")
    assert (problem_dir / "data_profile.png").is_file()
    assert (problem_dir / "histories.png").is_file()
    problem_dir = group_dir / problem_b.name.replace(" ", "_")
    assert (problem_dir / "data_profile.png").is_file()
    assert (problem_dir / "histories.png").is_file()
