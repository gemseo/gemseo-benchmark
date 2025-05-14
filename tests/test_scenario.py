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
"""Tests for the benchmarking scenario."""

from __future__ import annotations

import json
import os

import pytest

from gemseo_benchmark.scenario import Scenario


def test_inexistent_outputs_path(algorithms_configurations):
    """Check the handling of a nonexistent path of the outputs."""
    outputs_path = "/not/a/path/"
    with pytest.raises(
        NotADirectoryError,
        match=f"The path to the outputs directory does not exist: {outputs_path}.",
    ):
        Scenario([algorithms_configurations], outputs_path)


@pytest.mark.parametrize("n_processes", [1, 2])
@pytest.mark.parametrize("use_threading", [False, True])
@pytest.mark.parametrize("save_databases", [False, True])
def test_execute(
    algorithms_configurations,
    tmp_path,
    problems_group,
    save_databases,
    n_processes,
    use_threading,
):
    """Check the execution of a benchmarking scenario."""
    Scenario([algorithms_configurations], tmp_path).execute(
        [problems_group],
        save_databases=save_databases,
        n_processes=n_processes,
        use_threading=use_threading,
    )
    assert (tmp_path / "histories").is_dir()
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "report").is_dir()
    assert (tmp_path / "databases").is_dir() == save_databases


def test_report_overwrite(algorithms_configurations, tmp_path, problems_group) -> None:
    """Check that the report directory can be overwritten."""
    directory_path = tmp_path / "report"
    directory_path.mkdir()
    time = os.path.getmtime(directory_path)
    Scenario([algorithms_configurations], tmp_path).execute(
        [problems_group],
    )
    assert os.path.getmtime(directory_path) > time


def test_overlapping_algorithm_configurations(
    algorithms_configurations, tmp_path, problems_group, algorithm_configuration
) -> None:
    """Check the support of overlapping sets of algorithm configurations."""
    Scenario([algorithms_configurations, algorithms_configurations], tmp_path).execute(
        [problems_group], skip_report=True
    )
    with (tmp_path / "results.json").open() as file:
        data = json.load(file)

    # Check that only two result files have been generated.
    assert len(data[algorithm_configuration.name][problems_group.name]) == 2
