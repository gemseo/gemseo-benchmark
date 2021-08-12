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
from unittest.mock import Mock

import pytest
from report.report import Report


@pytest.fixture
def algo_name():
    return "SLSQP"


@pytest.fixture
def algos_specifications(algo_name):
    return {algo_name: dict()}


@pytest.fixture
def problem_name():
    return "A problem"


@pytest.fixture
def problem(problem_name):
    mock = Mock()
    mock.name = problem_name
    return mock


@pytest.fixture
def group(problem):
    mock = Mock()
    mock.name = "A group"
    mock.__iter__ = Mock(return_value=iter([problem]))
    return mock


@pytest.fixture
def problems_groups(group):
    return [group]


@pytest.fixture
def results(algo_name, problem_name, tmpdir):
    mock = Mock()
    mock.algorithms = [algo_name]
    mock.get_problems = Mock(return_value=[problem_name])
    return mock


def test_init_missing_algorithms(
        tmpdir, algo_name, algos_specifications, problems_groups, results
):
    """Check the initialization of the report with missing algorithms histories."""
    results.algorithms = ["Another algo"]
    with pytest.raises(
            ValueError, match="Missing histories for algorithm '{}'.".format(algo_name)
    ):
        Report(tmpdir, algos_specifications, problems_groups, results)


def test_init_missing_problems(
        tmpdir, algo_name, algos_specifications, problems_groups, results
):
    """Check the initialization of the report with missing problems histories."""
    results.get_problems = Mock(return_value=["Another problem"])
    with pytest.raises(
            ValueError,
            match="Missing histories for algorithm '{}' on problem 'A problem'.".format(
                algo_name
            )
    ):
        Report(tmpdir, algos_specifications, problems_groups, results)


def test_generate_report_sources(
        tmpdir, algos_specifications, problems_groups, results
):
    """Check the generation of the report sources."""
    report = Report(tmpdir, algos_specifications, problems_groups, results)
    report.generate_report(to_pdf=True)
    assert (tmpdir / "index.rst").isfile()
    assert (tmpdir / "algorithms.rst").isfile()
    assert (tmpdir / "problems_groups.rst").isfile()
    assert (tmpdir / "groups" / "A_group.rst").isfile()
    assert (tmpdir / "_build" / "html" / "index.html").isfile()
    assert (tmpdir / "_build" / "latex" / "benchmarking_report.pdf").isfile()


def test_retrieve_description(tmpdir, algos_specifications, problems_groups, results):
    """Check the retrieval of a GEMSEO algorithm description."""
    report = Report(tmpdir, algos_specifications, problems_groups, results)
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
    with open(tmpdir / "algorithms.rst") as file:
        contents = file.readlines()
    assert contents == ref_contents
