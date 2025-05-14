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

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from gemseo.problems.optimization.rastrigin import Rastrigin
from gemseo.utils.platform import PLATFORM_IS_WINDOWS
from numpy import array

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.benchmarker.benchmarker import Benchmarker
from gemseo_benchmark.problems.optimization_problem_configuration import (
    OptimizationBenchmarkingProblem,
)

if TYPE_CHECKING:
    from gemseo_benchmark.results.results import Results


@pytest.fixture(scope="module")
def rastrigin() -> OptimizationBenchmarkingProblem:
    """A problem configuration based on the 2-dimensional Rastrigin function."""
    return OptimizationBenchmarkingProblem(
        "Rastrigin", Rastrigin, [array([0.0, 0.1]), array([0.1, 0.0])]
    )


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
        f"Solving problem {index} "
        f"of problem configuration {rosenbrock.name} "
        f"with algorithm configuration {lbfgsb_configuration.name}."
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
        match=(
            f"The algorithm '{unknown_algorithm_configuration.algorithm_name}' "
            "is not available."
        ),
    ):
        Benchmarker(tmp_path, tmp_path / "results.json").execute(
            [rosenbrock], unknown_algorithms_configurations
        )


def test_is_problem_unsolved(tmp_path, rosenbrock, rastrigin, caplog):
    """Check the skipping of a problem."""
    results_path = tmp_path / "results.json"
    algo_config = AlgorithmConfiguration("L-BFGS-B")
    # Run the algorithm on the Rosenbrock problem configuration
    Benchmarker(tmp_path, results_path).execute(
        [rosenbrock], AlgorithmsConfigurations(algo_config)
    )
    # Run the algorithm on the Rastrigin problem configuration
    Benchmarker(tmp_path, results_path).execute(
        [rosenbrock, rastrigin], AlgorithmsConfigurations(algo_config)
    )
    assert (
        f"Skipping problem 1 of problem configuration {rosenbrock.name} for algorithm "
        f"configuration {algo_config.name}." in caplog.text
    )
    assert (
        f"Skipping problem 2 of problem configuration {rosenbrock.name} for algorithm "
        f"configuration {algo_config.name}." in caplog.text
    )


@pytest.mark.parametrize(
    ("n_processes", "use_threading"), [(1, False), (2, False), (2, True)]
)
def test_execution(results_root, rosenbrock, n_processes, use_threading):
    """Check the execution of the benchmarker."""
    results = Benchmarker(results_root).execute(
        [rosenbrock],
        AlgorithmsConfigurations(lbfgsb_configuration),
        n_processes=n_processes,
        use_threading=use_threading,
    )
    algo_pb_dir = results_root / lbfgsb_configuration.name / rosenbrock.name
    path = algo_pb_dir / f"{lbfgsb_configuration.name}.1.json"
    assert path.is_file()
    assert results.contains(lbfgsb_configuration.algorithm_name, rosenbrock.name, path)
    path = algo_pb_dir / f"{lbfgsb_configuration.name}.2.json"
    assert path.is_file()
    assert results.contains(lbfgsb_configuration.algorithm_name, rosenbrock.name, path)


def test_problem_specific_algorithm_options(results_root, rosenbrock):
    """Check problem-specific algorithm options."""
    Benchmarker(results_root).execute(
        [rosenbrock],
        AlgorithmsConfigurations(
            AlgorithmConfiguration(
                "L-BFGS-B",
                instance_algorithm_options={
                    "max_iter": lambda problem_configuration,
                    index: problem_configuration.dimension + index
                },
            )
        ),
    )
    path_base = (
        results_root
        / lbfgsb_configuration.name
        / rosenbrock.name
        / lbfgsb_configuration.name
    )
    with path_base.with_suffix(".1.json").open("r") as json_file_1:
        assert (
            json.load(json_file_1)["algorithm_configuration"]["algorithm_options"][
                "max_iter"
            ]
            == 2
        )

    with path_base.with_suffix(".2.json").open("r") as json_file_2:
        assert (
            json.load(json_file_2)["algorithm_configuration"]["algorithm_options"][
                "max_iter"
            ]
            == 3
        )


@pytest.mark.parametrize("n_processes", [1, 2])
@pytest.mark.parametrize("use_threading", [False, True])
def test_log_to_file(
    algorithms_configurations,
    algorithm_configuration,
    rosenbrock,
    n_processes,
    use_threading,
    tmp_path,
) -> None:
    """Check the logging of algorithms."""
    Benchmarker(tmp_path).execute(
        [rosenbrock],
        algorithms_configurations,
        n_processes=n_processes,
        use_threading=use_threading,
        save_log=True,
    )
    if use_threading:
        with (
            (tmp_path / "gemseo.log").open("r") as file,
            (Path(__file__).parent / "1.log").open("r") as reference1,
            (Path(__file__).parent / "2.log").open("r") as reference2,
        ):
            file_contents = file.read()
            for line in reference1.read().split("\n"):
                assert line in file_contents

            for line in reference2.read().split("\n"):
                assert line in file_contents
    else:
        if PLATFORM_IS_WINDOWS:
            # FIXME: Support logging to file when multiprocessing on Windows.
            return

        path_stem = (
            tmp_path
            / algorithm_configuration.name
            / rosenbrock.name
            / f"{algorithm_configuration.name}"
        )
        with (
            path_stem.with_suffix(".1.log").open("r") as file1,
            (Path(__file__).parent / "1.log").open("r") as reference1,
        ):
            assert reference1.read() in file1.read()

        with (
            path_stem.with_suffix(".2.log").open("r") as file2,
            (Path(__file__).parent / "2.log").open("r") as reference2,
        ):
            assert reference2.read() in file2.read()


@pytest.mark.parametrize("overwrite_histories", [False, True])
def test_overwrite_histories_results_update(
    tmp_path, rosenbrock, algorithms_configurations, overwrite_histories
) -> None:
    """Check that results are correctly updated when overwriting histories."""
    results_path = tmp_path / "results.json"
    history1_path = tmp_path / "SLSQP" / "Rosenbrock" / "SLSQP.1.json"
    history2_path = tmp_path / "SLSQP" / "Rosenbrock" / "SLSQP.2.json"
    benchmarker = Benchmarker(tmp_path, results_path)
    benchmarker.execute([rosenbrock], algorithms_configurations)
    with results_path.open() as results_file:
        data = json.load(results_file)
        assert data.keys() == {"SLSQP"}
        assert data["SLSQP"].keys() == {"Rosenbrock"}
        assert set(data["SLSQP"]["Rosenbrock"]) == {
            str(history1_path),
            str(history2_path),
        }

    results_time = results_path.stat().st_mtime
    history1_time = history1_path.stat().st_mtime
    history2_time = history2_path.stat().st_mtime

    benchmarker.execute([rosenbrock], algorithms_configurations, overwrite_histories)
    with results_path.open() as results_file:
        data = json.load(results_file)
        assert data.keys() == {"SLSQP"}
        assert data["SLSQP"].keys() == {"Rosenbrock"}
        assert set(data["SLSQP"]["Rosenbrock"]) == {
            str(history1_path),
            str(history2_path),
        }

    if overwrite_histories:
        assert results_path.stat().st_mtime > results_time
        assert history1_path.stat().st_mtime > history1_time
        assert history2_path.stat().st_mtime > history2_time
    else:
        assert results_path.stat().st_mtime == results_time
        assert history1_path.stat().st_mtime == history1_time
        assert history2_path.stat().st_mtime == history2_time


def test_worker_raised_exception(
    tmp_path, rosenbrock, algorithm_configuration, caplog
) -> None:
    """Check the case where a worker raised an exception."""
    problem = OptimizationBenchmarkingProblem(
        "Problem configuration", lambda: rosenbrock.create_problem()
    )
    Benchmarker(tmp_path).execute(
        [problem],
        AlgorithmsConfigurations(algorithm_configuration, name="Configuration"),
    )
    ((module, level, message),) = caplog.record_tuples
    assert module == "gemseo_benchmark.benchmarker.benchmarker"
    assert level == logging.WARNING
    assert message in {
        (
            "Solving problem 1 of problem configuration Problem configuration "
            f"for algorithm configuration {algorithm_configuration.name} "
            "raised: Can't pickle local object "
            "'test_worker_raised_exception.<locals>.<lambda>'"
        ),
        (
            "Solving problem 1 of problem configuration Problem configuration "
            f"for algorithm configuration {algorithm_configuration.name} "
            "raised: Can't get local object "
            "'test_worker_raised_exception.<locals>.<lambda>'"
        ),
    }
