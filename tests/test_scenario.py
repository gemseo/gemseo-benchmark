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

import pytest
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.scenario import Scenario


def test_inexistent_outputs_path(algorithms_configurations):
    """Check the handling of a nonexistent path of the outputs."""
    outputs_path = "/not/a/path/"
    with pytest.raises(
        NotADirectoryError,
        match=f"The path to the outputs directory does not exist: {outputs_path}.",
    ):
        Scenario([algorithms_configurations], outputs_path)


@pytest.mark.parametrize("number_of_processes", [1, 2])
@pytest.mark.parametrize("use_threading", [False, True])
@pytest.mark.parametrize("save_databases", [False, True])
@pytest.mark.parametrize("save_pseven_logs", [False, True])
def test_execute(
    algorithms_configurations,
    tmp_path,
    rosenbrock,
    save_databases,
    save_pseven_logs,
    number_of_processes,
    use_threading,
):
    """Check the execution of a benchmarking scenario."""
    Scenario([algorithms_configurations], tmp_path).execute(
        [ProblemsGroup("Rosenbrock", [rosenbrock])],
        save_databases=save_databases,
        save_pseven_logs=save_pseven_logs,
        number_of_processes=number_of_processes,
        use_threading=use_threading,
    )
    assert (tmp_path / "histories").is_dir()
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "report").is_dir()
    assert (tmp_path / "databases").is_dir() == save_databases
    assert not (tmp_path / "pseven_logs").is_dir()


@pytest.mark.skipif(
    not OptimizersFactory().is_available("PSEVEN"), reason="pSeven is not available."
)
@pytest.mark.parametrize("save_pseven_logs", [False, True])
def test_execute_pseven(
    algorithms_configurations, tmp_path, rosenbrock, save_pseven_logs
):
    """Check the execution of a benchmarking scenario including pSeven."""
    Scenario(
        [AlgorithmsConfigurations(AlgorithmConfiguration("PSEVEN"))], tmp_path
    ).execute(
        [ProblemsGroup("Rosenbrock", [rosenbrock])], save_pseven_logs=save_pseven_logs
    )
    assert (tmp_path / "pseven_logs").is_dir() == save_pseven_logs
