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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the benchmarker."""
from __future__ import annotations

import pytest
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.rastrigin import Rastrigin
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.benchmarker.benchmarker import Benchmarker
from gemseo_benchmark.problems.problem import Problem
from gemseo_benchmark.results.results import Results
from numpy import array


@pytest.fixture(scope="module")
def rastrigin() -> Problem:
    """A benchmarking problem based on the 2-dimensional Rastrigin function."""
    return Problem("Rastrigin", Rastrigin, [array([0.0, 0.1]), array([0.1, 0.0])])


lbfgsb_configuration = AlgorithmConfiguration("L-BFGS-B")


@pytest.fixture(scope="module")
def lbfgsb_results(results_root, rosenbrock) -> Results:
    """The results of L-BFGS-B on the Rosenbrock function."""
    return Benchmarker(results_root, results_root / "results.json").execute(
        [rosenbrock], AlgorithmsConfigurations(lbfgsb_configuration)
    )


@pytest.mark.parametrize("index", [1, 2])
def test_save_history(results_root, rosenbrock, lbfgsb_results, index):
    """Check the saving of performance histories."""
    algo_pb_dir = results_root / lbfgsb_configuration.name / rosenbrock.name
    path = algo_pb_dir / f"{lbfgsb_configuration.name}.{index}.json"
    assert path.is_file()
    assert lbfgsb_results.contains(
        lbfgsb_configuration.algorithm_name, rosenbrock.name, path
    )
    assert (
        f"Solving instance {index} of problem {rosenbrock.name} with algorithm "
        f"configuration {lbfgsb_configuration.name}."
    )


def test_save_database(tmp_path, rosenbrock):
    """Check the saving of optimization databases."""
    algo_config = AlgorithmConfiguration("L-BFGS-B")
    Benchmarker(tmp_path, tmp_path / "results.json", tmp_path).execute(
        [rosenbrock], AlgorithmsConfigurations(algo_config)
    )
    algo_pb_dir = tmp_path / algo_config.name / rosenbrock.name
    assert (algo_pb_dir / f"{algo_config.name}.1.h5").is_file()
    assert (algo_pb_dir / f"{algo_config.name}.2.h5").is_file()


def test_unavailable_algorithm(
    tmp_path,
    rosenbrock,
    unknown_algorithm_configuration,
    unknown_algorithms_configurations,
):
    """Check the handling of an unavailable algorithm."""
    with pytest.raises(
        ValueError,
        match="The algorithm is not available: "
        f"{unknown_algorithm_configuration.algorithm_name}.",
    ):
        Benchmarker(tmp_path, tmp_path / "results.json").execute(
            [rosenbrock], unknown_algorithms_configurations
        )


def test___skip_instance(tmp_path, rosenbrock, rastrigin, caplog):
    """Check the skipping of an optimization."""
    results_path = tmp_path / "results.json"
    algo_config = AlgorithmConfiguration("L-BFGS-B")
    # Run the algorithm on the Rosenbrock problem
    Benchmarker(tmp_path, results_path).execute(
        [rosenbrock], AlgorithmsConfigurations(algo_config)
    )
    # Run the algorithm on the Rastrigin problem
    Benchmarker(tmp_path, results_path).execute(
        [rosenbrock, rastrigin], AlgorithmsConfigurations(algo_config)
    )
    assert (
        f"Skipping instance 1 of problem {rosenbrock.name} for algorithm "
        f"configuration {algo_config.name}." in caplog.text
    )
    assert (
        f"Skipping instance 2 of problem {rosenbrock.name} for algorithm "
        f"configuration {algo_config.name}." in caplog.text
    )


@pytest.mark.skipif(
    not OptimizersFactory().is_available("PSEVEN"), reason="pSeven is not available."
)
def test___set_pseven_log_file(tmp_path, rosenbrock):
    """Check the setting of the pSeven log file."""
    results_path = tmp_path / "results.json"
    algo_config = AlgorithmConfiguration("PSEVEN")
    Benchmarker(tmp_path, results_path, pseven_logs_path=tmp_path).execute(
        [rosenbrock], AlgorithmsConfigurations(algo_config)
    )
    algo_pb_dir = tmp_path / algo_config.name / rosenbrock.name
    assert (algo_pb_dir / f"{algo_config.name}.1.txt").is_file()
    assert (algo_pb_dir / f"{algo_config.name}.2.txt").is_file()


@pytest.mark.parametrize(
    ["number_of_processes", "use_threading"], [(1, False), (2, False), (2, True)]
)
def test_execution(results_root, rosenbrock, number_of_processes, use_threading):
    """Check the execution of the benchmarker."""
    results = Benchmarker(results_root).execute(
        [rosenbrock],
        AlgorithmsConfigurations(lbfgsb_configuration),
        number_of_processes=number_of_processes,
        use_threading=use_threading,
    )
    algo_pb_dir = results_root / lbfgsb_configuration.name / rosenbrock.name
    path = algo_pb_dir / f"{lbfgsb_configuration.name}.1.json"
    assert path.is_file()
    assert results.contains(lbfgsb_configuration.algorithm_name, rosenbrock.name, path)
    path = algo_pb_dir / f"{lbfgsb_configuration.name}.1.json"
    assert path.is_file()
    assert results.contains(lbfgsb_configuration.algorithm_name, rosenbrock.name, path)
