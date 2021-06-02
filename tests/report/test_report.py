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

from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import zeros
from pytest import raises

from data_profiles.target_values import TargetValues
from problems.problem import Problem
from problems.problems_group import ProblemsGroup
from report.report import Report


def get_report_args(histories_dir):
    """Return the arguments for the report initialization.

    Args:
        histories_dir: Path to the histories directory.

    Returns:
        The algorithms and their options, the groups of problems, the histories paths.
    """
    algorithms = {"An algo": dict()}
    targets = TargetValues(list(range(1000, 0, -100)))
    a_problem = Problem("A problem", Rosenbrock, [zeros(2)], targets)
    problems_groups = [ProblemsGroup("A group", [a_problem], "A description")]
    targets.to_file(str(histories_dir / "history.json"))
    histories_path = {"An algo": {"A problem": [str(histories_dir / "history.json")]}}
    return algorithms, problems_groups, histories_path


def test_init(tmpdir):
    """Check the initialization of the report."""
    algorithms, problems_groups, histories_path = get_report_args(tmpdir)
    with raises(ValueError):
        Report(tmpdir, {"tata": dict()}, problems_groups, histories_path)
    with raises(ValueError):
        Report(tmpdir, algorithms, problems_groups, {"tata": histories_path["An algo"]})


def test_generate_report_sources(tmpdir):
    """Check the generation of the report sources."""
    algorithms, problems_groups, histories_path = get_report_args(tmpdir)
    report = Report(tmpdir, algorithms, problems_groups, histories_path)
    report.generate_report_sources()
    assert (tmpdir / "index.rst").isfile()
    assert (tmpdir / "algorithms.rst").isfile()
    assert (tmpdir / "problems_groups.rst").isfile()
    assert (tmpdir / "groups" / "A_group.rst").isfile()
    assert (tmpdir / "_build" / "html" / "index.html").isfile()
