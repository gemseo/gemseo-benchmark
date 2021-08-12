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
from data_profiles.target_values import TargetValues
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.utils.py23_compat import Path
from numpy import zeros
from problems.problem import Problem
from problems.problems_group import ProblemsGroup
from report.report import Report


def get_report_args(
        histories_dir,  # type: Path
        algo_name="An algo",  # type: str
):
    """Return the arguments for the report initialization.

    Args:
        histories_dir: Path to the histories directory.
        algo_name: The name of the algorithm.
            If None, defaults to "An algo".

    Returns:
        The algorithms and their options, the groups of problems, the histories paths.
    """
    algos_specifications = {algo_name: dict()}
    targets = TargetValues(list(range(1000, 0, -100)))
    a_problem = Problem("A problem", Rosenbrock, [zeros(2)], targets)
    problems_groups = [ProblemsGroup("A group", [a_problem], "A description")]
    targets.to_file(str(histories_dir / "history.json"))
    histories_path = {algo_name: {"A problem": [str(histories_dir / "history.json")]}}
    # FIXME: replace with a Results mock
    return algos_specifications, problems_groups, histories_path


def test_init_missing_algorithms(tmpdir):
    """Check the initialization of the report with missing algorithms histories."""
    algos_specs, problems_groups, _ = get_report_args(tmpdir)
    results = Mock()
    results.algorithms = ["Another algo"]
    with pytest.raises(ValueError, match="Missing histories for algorithm 'An algo'"):
        Report(tmpdir, algos_specs, problems_groups, results)


def test_init_missing_problems(tmpdir):
    """Check the initialization of the report with missing problems histories."""
    algos_specs, problems_groups, _ = get_report_args(tmpdir)
    results = Mock()
    results.algorithms = ["An algo"]
    results.get_problems = Mock(return_value=["Another problem"])
    with pytest.raises(
            ValueError,
            match="Missing histories for algorithm 'An algo' on problem 'A problem'"
    ):
        Report(tmpdir, algos_specs, problems_groups, results)


def test_generate_report_sources(tmpdir):
    """Check the generation of the report sources."""
    algorithms, problems_groups, histories_path = get_report_args(tmpdir)
    report = Report(tmpdir, algorithms, problems_groups, histories_path)
    report.generate_report(to_pdf=True)
    assert (tmpdir / "index.rst").isfile()
    assert (tmpdir / "algorithms.rst").isfile()
    assert (tmpdir / "problems_groups.rst").isfile()
    assert (tmpdir / "groups" / "A_group.rst").isfile()
    assert (tmpdir / "_build" / "html" / "index.html").isfile()
    assert (tmpdir / "_build" / "latex" / "benchmarking_report.pdf").isfile()


def test_retrieve_description(tmpdir):
    """Check the retrieval of a Gemseo algorithm description."""
    algorithms, problems_groups, histories_path = get_report_args(tmpdir, "SLSQP")
    report = Report(tmpdir, algorithms, problems_groups, histories_path)
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
