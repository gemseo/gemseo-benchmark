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
from gemseo.utils.platform import PLATFORM_IS_WINDOWS

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.benchmarker.benchmarker import Benchmarker
from gemseo_benchmark.problems.mda_problem_configuration import MDAProblemConfiguration
from gemseo_benchmark.problems.mdo_problem_configuration import MDOProblemConfiguration
from gemseo_benchmark.problems.optimization_problem_configuration import (
    OptimizationProblemConfiguration,
)

if TYPE_CHECKING:
    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )
    from gemseo_benchmark.results.results import Results


@pytest.fixture(scope="module")
def optimization_algorithm_configuration() -> AlgorithmConfiguration:
    """An algorithm configuration for optimization."""
    return AlgorithmConfiguration("L-BFGS-B")


def get_results(
    results_root: Path,
    algorithm_configuration: AlgorithmConfiguration,
    problem_configuration: BaseProblemConfiguration,
) -> Results:
    return Benchmarker(results_root, results_root / "results.json").execute(
        [problem_configuration], AlgorithmsConfigurations(algorithm_configuration)
    )


@pytest.fixture(scope="module")
def optimization_results(
    results_root, rosenbrock, optimization_algorithm_configuration
) -> Results:
    """The results of an optimization benchmarking."""
    return get_results(results_root, optimization_algorithm_configuration, rosenbrock)


@pytest.fixture(scope="module")
def mda_histories_dir(tmp_path_factory) -> Path:
    """The directory containing the MDA performance histories."""
    return tmp_path_factory.mktemp("mda_results")


@pytest.fixture(scope="module")
def mda_results(
    mda_algorithm_configuration, mda_problem_configuration, mda_histories_dir
) -> Results:
    """The results of a multidisciplinary analysis benchmarking."""
    return get_results(
        mda_histories_dir, mda_algorithm_configuration, mda_problem_configuration
    )


@pytest.fixture(scope="module")
def mdo_histories_dir(tmp_path_factory) -> Path:
    """The directory containing the MDO performance histories."""
    return tmp_path_factory.mktemp("mdo_results")


@pytest.fixture(scope="module")
def mdo_results(
    mdo_algorithm_configuration, mdo_problem_configuration, mdo_histories_dir
) -> Results:
    """The results of a multidisciplinary optimization benchmarking."""
    return get_results(
        mdo_histories_dir, mdo_algorithm_configuration, mdo_problem_configuration
    )


@pytest.mark.parametrize(
    (
        "algorithm_configuration",
        "problem_configuration",
        "results",
        "histories_dir",
    ),
    [
        (
            "optimization_algorithm_configuration",
            "rosenbrock",
            "optimization_results",
            "results_root",
        ),
        (
            "mda_algorithm_configuration",
            "mda_problem_configuration",
            "mda_results",
            "mda_histories_dir",
        ),
        (
            "mdo_algorithm_configuration",
            "mdo_problem_configuration",
            "mdo_results",
            "mdo_histories_dir",
        ),
    ],
)
@pytest.mark.parametrize("index", [1, 2])
def test_save_history(
    algorithm_configuration,
    problem_configuration,
    index,
    results,
    histories_dir,
    request,
) -> None:
    """Check the saving of performance histories."""
    algo_config = request.getfixturevalue(algorithm_configuration)
    pb_config = request.getfixturevalue(problem_configuration)
    reslts = request.getfixturevalue(results)
    path = (
        request.getfixturevalue(histories_dir)
        / algo_config.name
        / pb_config.name.replace(" ", "_")
        / f"{algo_config.name}.{index}.json"
    )
    assert path.is_file()
    assert reslts.contains(algo_config.algorithm_name, pb_config.name, path)


configurations = pytest.mark.parametrize(
    ("algorithm_configuration", "problem_configuration"),
    [
        ("optimization_algorithm_configuration", "rosenbrock"),
        ("mda_algorithm_configuration", "mda_problem_configuration"),
        ("mdo_algorithm_configuration", "mdo_problem_configuration"),
    ],
)


@configurations
def test_save_data(
    tmp_path, algorithm_configuration, problem_configuration, request
) -> None:
    """Check the saving of algorithm data."""
    algo_config = request.getfixturevalue(algorithm_configuration)
    pb_config = request.getfixturevalue(problem_configuration)
    Benchmarker(tmp_path, tmp_path / "results.json", tmp_path).execute(
        [pb_config], AlgorithmsConfigurations(algo_config)
    )
    histories_dir = tmp_path / algo_config.name / pb_config.name.replace(" ", "_")
    assert (histories_dir / f"{algo_config.name}.1.h5").is_file()
    assert (histories_dir / f"{algo_config.name}.2.h5").is_file()


@configurations
def test_unavailable_algorithm(
    tmp_path,
    algorithm_configuration,
    problem_configuration,
    unknown_algorithm_configuration,
    request,
) -> None:
    """Check the handling of an unavailable algorithm."""
    with pytest.raises(
        ValueError,
        match=(
            f"The algorithm '{unknown_algorithm_configuration.algorithm_name}' "
            "is not available."
        ),
    ):
        Benchmarker(tmp_path, tmp_path / "results.json").execute(
            [request.getfixturevalue(problem_configuration)],
            AlgorithmsConfigurations(
                unknown_algorithm_configuration,
                request.getfixturevalue(algorithm_configuration),
            ),
        )


@configurations
def test_is_problem_unsolved(
    tmp_path, algorithm_configuration, problem_configuration, caplog, request
) -> None:
    """Check the skipping of a problem."""
    results_path = tmp_path / "results.json"
    algo_config = request.getfixturevalue(algorithm_configuration)
    pb_config = request.getfixturevalue(problem_configuration)
    Benchmarker(tmp_path, results_path).execute(
        [pb_config], AlgorithmsConfigurations(algo_config)
    )
    Benchmarker(tmp_path, results_path).execute(
        [pb_config], AlgorithmsConfigurations(algo_config)
    )
    assert (
        f"Skipping problem 1 of problem configuration {pb_config.name} for algorithm "
        f"configuration {algo_config.name}." in caplog.text
    )
    assert (
        f"Skipping problem 2 of problem configuration {pb_config.name} for algorithm "
        f"configuration {algo_config.name}." in caplog.text
    )


@configurations
@pytest.mark.parametrize("n_processes", [1, 2])
@pytest.mark.parametrize("use_threading", [False, True])
def test_execution(
    results_root,
    algorithm_configuration,
    problem_configuration,
    n_processes,
    use_threading,
    request,
) -> None:
    """Check the execution of the benchmarker."""
    algo_config = request.getfixturevalue(algorithm_configuration)
    pb_config = request.getfixturevalue(problem_configuration)
    results = Benchmarker(results_root).execute(
        [pb_config],
        AlgorithmsConfigurations(algo_config),
        n_processes=n_processes,
        use_threading=use_threading,
    )
    histories_dir = results_root / algo_config.name / pb_config.name.replace(" ", "_")
    path = histories_dir / f"{algo_config.name}.1.json"
    assert path.is_file()
    assert results.contains(algo_config.algorithm_name, pb_config.name, path)
    path = histories_dir / f"{algo_config.name}.2.json"
    assert path.is_file()
    assert results.contains(algo_config.algorithm_name, pb_config.name, path)


@pytest.mark.parametrize(
    ("algorithm_configuration", "problem_configuration", "option_name"),
    [
        ("optimization_algorithm_configuration", "rosenbrock", "max_iter"),
        ("mda_algorithm_configuration", "mda_problem_configuration", "max_mda_iter"),
        ("mdo_algorithm_configuration", "mdo_problem_configuration", "max_iter"),
    ],
)
def test_problem_specific_algorithm_options(
    results_root, algorithm_configuration, problem_configuration, option_name, request
) -> None:
    """Check problem-specific algorithm options."""
    algo_config = request.getfixturevalue(algorithm_configuration)
    pb_config = request.getfixturevalue(problem_configuration)
    Benchmarker(results_root).execute(
        [pb_config],
        AlgorithmsConfigurations(
            AlgorithmConfiguration(
                algo_config.algorithm_name,
                instance_algorithm_options={
                    option_name: lambda config, index: config.dimension + index
                },
            )
        ),
    )
    path_base = (
        results_root
        / algo_config.name
        / pb_config.name.replace(" ", "_")
        / algo_config.name
    )
    with path_base.with_suffix(".1.json").open("r") as json_file_1:
        assert (
            json.load(json_file_1)["algorithm_configuration"]["algorithm_options"][
                option_name
            ]
            == 2
        )

    with path_base.with_suffix(".2.json").open("r") as json_file_2:
        assert (
            json.load(json_file_2)["algorithm_configuration"]["algorithm_options"][
                option_name
            ]
            == 3
        )


@configurations
@pytest.mark.parametrize("n_processes", [1, 2])
@pytest.mark.parametrize("use_threading", [False, True])
def test_log_to_file(
    algorithm_configuration,
    problem_configuration,
    n_processes,
    use_threading,
    tmp_path,
    request,
) -> None:
    """Check the logging of algorithms."""
    algo_config = request.getfixturevalue(algorithm_configuration)
    pb_config = request.getfixturevalue(problem_configuration)
    Benchmarker(tmp_path).execute(
        [pb_config],
        AlgorithmsConfigurations(algo_config),
        n_processes=n_processes,
        use_threading=use_threading,
        save_log=True,
    )
    reference1_path = Path(__file__).parent / f"{algo_config.name}.1.log"
    reference2_path = Path(__file__).parent / f"{algo_config.name}.2.log"
    if use_threading:
        with (
            (tmp_path / "gemseo.log").open("r") as file,
            reference1_path.open("r") as reference1,
            reference2_path.open("r") as reference2,
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

        path_base = (
            tmp_path
            / algo_config.name
            / pb_config.name.replace(" ", "_")
            / f"{algo_config.name}"
        )
        with (
            path_base.with_suffix(".1.log").open("r") as file1,
            reference1_path.open("r") as reference1,
        ):
            assert reference1.read() in file1.read()

        with (
            path_base.with_suffix(".2.log").open("r") as file2,
            reference2_path.open("r") as reference2,
        ):
            assert reference2.read() in file2.read()


@configurations
@pytest.mark.parametrize("overwrite_histories", [False, True])
def test_overwrite_histories_results_update(
    tmp_path,
    algorithm_configuration,
    problem_configuration,
    overwrite_histories,
    request,
) -> None:
    """Check that results are correctly updated when overwriting histories."""
    algo_config = request.getfixturevalue(algorithm_configuration)
    algo_config_name = algo_config.name
    pb_config = request.getfixturevalue(problem_configuration)
    pb_config_name = pb_config.name
    results_path = tmp_path / "results.json"
    history1_path = (
        tmp_path
        / algo_config_name
        / pb_config_name.replace(" ", "_")
        / f"{algo_config_name}.1.json"
    )
    history2_path = (
        tmp_path
        / algo_config_name
        / pb_config_name.replace(" ", "_")
        / f"{algo_config_name}.2.json"
    )
    benchmarker = Benchmarker(tmp_path, results_path)
    benchmarker.execute([pb_config], AlgorithmsConfigurations(algo_config))
    with results_path.open() as results_file:
        data = json.load(results_file)
        assert data.keys() == {algo_config_name}
        assert data[algo_config_name].keys() == {pb_config_name}
        assert set(data[algo_config_name][pb_config_name]) == {
            str(history1_path),
            str(history2_path),
        }

    results_time = results_path.stat().st_mtime
    history1_time = history1_path.stat().st_mtime
    history2_time = history2_path.stat().st_mtime

    benchmarker.execute(
        [pb_config], AlgorithmsConfigurations(algo_config), overwrite_histories
    )
    with results_path.open() as results_file:
        data = json.load(results_file)
        assert data.keys() == {algo_config_name}
        assert data[algo_config_name].keys() == {pb_config_name}
        assert set(data[algo_config_name][pb_config_name]) == {
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


@pytest.fixture(scope="module")
def ill_optimization_problem_configuration(
    rosenbrock,
) -> OptimizationProblemConfiguration:
    """An ill optimization problem configuration."""
    return OptimizationProblemConfiguration(
        rosenbrock.name, lambda: rosenbrock.create_problem()
    )


@pytest.fixture(scope="module")
def ill_mda_problem_configuration(
    multidisciplinary_variable_space,
    mda_problem_configuration,
) -> MDAProblemConfiguration:
    """An ill multidisciplinary analysis problem configuration."""
    return MDAProblemConfiguration(
        mda_problem_configuration.name,
        lambda algorithm_configuration: mda_problem_configuration.create_problem(
            algorithm_configuration
        ),
        multidisciplinary_variable_space,
    )


@pytest.fixture(scope="module")
def ill_mdo_problem_configuration(
    multidisciplinary_variable_space,
    mdo_problem_configuration,
) -> MDOProblemConfiguration:
    """An ill multidisciplinary optimization problem configuration."""
    return MDOProblemConfiguration(
        mdo_problem_configuration.name,
        lambda algorithm_configuration: mdo_problem_configuration.create_problem(
            algorithm_configuration
        ),
        multidisciplinary_variable_space,
        True,
        0,
    )


@pytest.mark.parametrize(
    ("algorithm_configuration", "ill_problem_configuration"),
    [
        (
            "optimization_algorithm_configuration",
            "ill_optimization_problem_configuration",
        ),
        ("mda_algorithm_configuration", "ill_mda_problem_configuration"),
        ("mdo_algorithm_configuration", "ill_mdo_problem_configuration"),
    ],
)
def test_worker_raised_exception(
    tmp_path, algorithm_configuration, ill_problem_configuration, caplog, request
) -> None:
    """Check the case where a worker raised an exception."""
    algo_config = request.getfixturevalue(algorithm_configuration)
    pb_config = request.getfixturevalue(ill_problem_configuration)
    Benchmarker(tmp_path).execute([pb_config], AlgorithmsConfigurations(algo_config))
    ((module, level, message),) = caplog.record_tuples
    assert module == "gemseo_benchmark.benchmarker.benchmarker"
    assert level == logging.WARNING
    assert message in {
        (
            f"Solving problem 1 of problem configuration {pb_config.name} "
            f"for algorithm configuration {algo_config.name} "
            f"raised: Can't {verb} local object "
            f"'{ill_problem_configuration}.<locals>.<lambda>'"
        )
        for verb in {"get", "pickle"}
    }
