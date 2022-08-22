# -*- coding: utf-8 -*-
# Copyright 2022 IRT Saint Exupéry, https://www.irt-saintexupery.com
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

import pytest
from numpy import array

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.rastrigin import Rastrigin
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import \
    AlgorithmsConfigurations
from gemseo_benchmark.benchmarker import Benchmarker
from gemseo_benchmark.problems.problem import Problem


@pytest.fixture(scope="module")
def rosenbrock() -> Problem:
    """A benchmarking problem based on the 2-dimensional Rosenbrock function."""
    return Problem("Rosenbrock", Rosenbrock, [array([0.0, 1.0]), array([1.0, 0.0])])


@pytest.fixture(scope="module")
def rastrigin() -> Problem:
    """A benchmarking problem based on the 2-dimensional Rastrigin function."""
    return Problem("Rastrigin", Rastrigin, [array([0.0, 0.1]), array([0.1, 0.0])])


def test_save_history(tmp_path, rosenbrock):
    """Check the saving of performance histories."""
    results_path = tmp_path / "results.json"
    algo_config = AlgorithmConfiguration("L-BFGS-B")
    results = Benchmarker(tmp_path, results_path).execute(
        [rosenbrock], AlgorithmsConfigurations(algo_config)
    )
    algo_pb_dir = tmp_path / algo_config.name / rosenbrock.name
    path = algo_pb_dir / f"{algo_config.name}.1.json"
    assert path.is_file()
    assert results.contains(algo_config.algorithm_name, rosenbrock.name, path)
    assert f"Solving instance 1 of problem {rosenbrock.name} with algorithm " \
           f"configuration {algo_config.name}."
    path = algo_pb_dir / f"{algo_config.name}.2.json"
    assert path.is_file()
    assert results.contains(algo_config.algorithm_name, rosenbrock.name, path)
    assert f"Solving instance 2 of problem {rosenbrock.name} with algorithm " \
           f"configuration {algo_config.name}."


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
        tmp_path, rosenbrock, unknown_algorithm_configuration,
        unknown_algorithms_configurations
):
    """Check the handling of an unavailable algorithm."""
    with pytest.raises(
            ValueError, match="The algorithm is not available: "
                              f"{unknown_algorithm_configuration.algorithm_name}."
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
    assert f"Skipping instance 1 of problem {rosenbrock.name} for algorithm " \
           f"configuration {algo_config.name}." in caplog.text
    assert f"Skipping instance 2 of problem {rosenbrock.name} for algorithm " \
           f"configuration {algo_config.name}." in caplog.text


@pytest.mark.skipif(
    not OptimizersFactory().is_available("PSEVEN"), reason="pSeven is not available."
)
def test___set_pseven_log_file(tmp_path, rosenbrock):
    """Check the setting of the pSeven log file."""
    results_path = tmp_path / "results.json"
    algo_config = AlgorithmConfiguration("PSEVEN")
    Benchmarker(tmp_path, results_path, pseven_dir=tmp_path).execute(
        [rosenbrock], AlgorithmsConfigurations(algo_config)
    )
    algo_pb_dir = tmp_path / algo_config.name / rosenbrock.name
    assert (algo_pb_dir / f"{algo_config.name}.1.txt").is_file()
    assert (algo_pb_dir / f"{algo_config.name}.2.txt").is_file()
