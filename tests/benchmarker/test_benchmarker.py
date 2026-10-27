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
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from gemseo.optimization import L_BFGS_B_Settings
from gemseo.optimization import TNC_Settings
from gemseo.problem.optimization.power_2 import Power2
from gemseo.util.platform import platform_is_windows
from gemseo.util.testing.helper import assert_exception

from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.benchmarker.benchmarker import Benchmarker
from gemseo_benchmark.benchmarker.benchmarking_error import BenchmarkingError
from gemseo_benchmark.problems.mda_problem_configuration import MDAProblemConfiguration
from gemseo_benchmark.problems.mdo_problem_configuration import MDOProblemConfiguration
from gemseo_benchmark.problems.optimization_problem_configuration import (
    OptimizationProblemConfiguration,
)
from gemseo_benchmark.results.results import Results

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gemseo_benchmark.problems.base_problem_configuration import (
        BaseProblemConfiguration,
    )


@pytest.fixture(scope="module")
def optimization_algorithm_configuration() -> AlgorithmConfiguration:
    """An algorithm configuration for optimization."""
    return AlgorithmConfiguration(L_BFGS_B_Settings())


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


def test_skipping_logged_to_file(tmp_wd, rosenbrock) -> None:
    """Check that the skipped problems are logged to the file shared by threads."""
    results_path = tmp_wd / "results.json"
    algorithm_configuration = AlgorithmConfiguration(L_BFGS_B_Settings())
    for _ in range(2):
        Benchmarker(tmp_wd, results_path).execute(
            [rosenbrock],
            AlgorithmsConfigurations(algorithm_configuration),
            use_threading=True,
            save_log=True,
        )

    log = (tmp_wd / "gemseo.log").read_text()
    for index in (1, 2):
        assert (
            f"Skipping problem {index} of problem configuration {rosenbrock.name} "
            f"for algorithm configuration {algorithm_configuration.name}."
        ) in log


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
    settings_class = type(algo_config.algorithm_settings)
    Benchmarker(results_root).execute(
        [pb_config],
        AlgorithmsConfigurations(
            AlgorithmConfiguration(
                settings_class(),
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
            json.load(json_file_1)["algorithm_configuration"]["algorithm_settings"][
                option_name
            ]
            == 2
        )

    with path_base.with_suffix(".2.json").open("r") as json_file_2:
        assert (
            json.load(json_file_2)["algorithm_configuration"]["algorithm_settings"][
                option_name
            ]
            == 3
        )


def raise_index_error(problem_configuration, index):
    """Raise an `IndexError` for the second problem.

    Args:
        problem_configuration: The problem configuration.
        index: The 0-based index of the problem.

    Returns:
        The maximum number of iterations.

    Raises:
        IndexError: For the second problem.
    """
    if index == 1:
        msg = "There is no value for the second problem."
        raise IndexError(msg)

    return 5


@pytest.mark.parametrize(
    "get_max_iter",
    [lambda config, index: "x" if index == 1 else 5, raise_index_error],
    ids=["invalid_value", "callable_error"],
)
def test_invalid_problem_specific_algorithm_option(
    tmp_wd, rosenbrock, get_max_iter, snapshot
) -> None:
    """Check that an invalid problem-specific option stops before any execution.

    The option is invalid when its value is invalid
    or when the callable computing it raises an error.
    """
    results_path = tmp_wd / "results.json"
    algorithm_configuration = AlgorithmConfiguration(
        L_BFGS_B_Settings(),
        "L-BFGS-B",
        instance_algorithm_options={"max_iter": get_max_iter},
    )
    loggers = (logging.getLogger(Benchmarker.__module__), logging.getLogger("gemseo"))
    handlers = [list(logger.handlers) for logger in loggers]
    with assert_exception(ValueError, snapshot):
        Benchmarker(tmp_wd, results_path).execute(
            [rosenbrock],
            AlgorithmsConfigurations(algorithm_configuration),
            use_threading=True,
            save_log=True,
        )

    assert not results_path.exists()
    assert not (tmp_wd / "L-BFGS-B").exists()
    assert [list(logger.handlers) for logger in loggers] == handlers


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
        if platform_is_windows:
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


ill_configurations = pytest.mark.parametrize(
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


@ill_configurations
def test_worker_raised_exception(
    tmp_path, algorithm_configuration, ill_problem_configuration, caplog, request
) -> None:
    """Check the case where a worker raised an exception, with the error raised."""
    algo_config = request.getfixturevalue(algorithm_configuration)
    pb_config = request.getfixturevalue(ill_problem_configuration)
    with pytest.raises(
        BenchmarkingError,
        match=re.escape("1 of 1 execution raised an exception"),
    ) as error_info:
        Benchmarker(tmp_path).execute(
            [pb_config], AlgorithmsConfigurations(algo_config), raise_errors=True
        )

    # The error lists the execution that failed, and is caused by its exception.
    (description, exception) = next(iter(error_info.value.exceptions.items()))
    assert description == (
        f"Solving problem 1 of problem configuration {pb_config.name} "
        f"for algorithm configuration {algo_config.name}"
    )
    assert f"- {description} raised: " in str(error_info.value)
    assert error_info.value.__cause__ is exception

    # The failure is also logged as an error, with its traceback, when it occurs.
    ((module, level, message),) = caplog.record_tuples
    assert module == "gemseo_benchmark.benchmarker.benchmarker"
    assert level == logging.ERROR
    assert caplog.records[0].exc_info[1] is exception
    assert message in {
        (
            f"Solving problem 1 of problem configuration {pb_config.name} "
            f"for algorithm configuration {algo_config.name} "
            f"raised: Can't {verb} local object "
            f"'{ill_problem_configuration}.<locals>.<lambda>'"
        )
        for verb in {"get", "pickle"}
    }


@ill_configurations
def test_worker_raised_exception_not_raised(
    tmp_path, algorithm_configuration, ill_problem_configuration, caplog, request
) -> None:
    """Check the case where a worker raised an exception, without raising the error."""
    algo_config = request.getfixturevalue(algorithm_configuration)
    pb_config = request.getfixturevalue(ill_problem_configuration)
    results = Benchmarker(tmp_path).execute(
        [pb_config], AlgorithmsConfigurations(algo_config)
    )
    assert results.algorithms == []

    # The failure is logged as an error, with its traceback, then summarized.
    (record, summary) = caplog.records
    assert record.levelno == summary.levelno == logging.ERROR
    assert record.exc_info is not None
    assert record.getMessage().startswith(
        f"Solving problem 1 of problem configuration {pb_config.name} "
        f"for algorithm configuration {algo_config.name} raised: "
    )
    assert summary.getMessage().startswith("1 of 1 execution raised an exception")


def create_failing_benchmarking(
    tmp_path: Path,
    rosenbrock: OptimizationProblemConfiguration,
    algorithm_configuration: AlgorithmConfiguration,
) -> tuple[
    Benchmarker,
    list[OptimizationProblemConfiguration],
    AlgorithmsConfigurations,
    Path,
]:
    """Create a benchmarking where the execution on one problem configuration fails.

    Args:
        tmp_path: The path to the working directory.
        rosenbrock: The problem configuration that succeeds.
        algorithm_configuration: The algorithm configuration.

    Returns:
        The benchmarker, the problem configurations,
        the algorithm configurations and the path to the results file.
    """
    ill_problem_configuration = OptimizationProblemConfiguration(
        "Ill problem", lambda: rosenbrock.create_problem()
    )
    results_path = tmp_path / "results.json"
    return (
        Benchmarker(tmp_path, results_path),
        [ill_problem_configuration, rosenbrock],
        AlgorithmsConfigurations(algorithm_configuration),
        results_path,
    )


def assert_succeeded_executions_saved(
    results_path: Path,
    rosenbrock: OptimizationProblemConfiguration,
    algorithm_configuration: AlgorithmConfiguration,
) -> None:
    """Check that the performance histories of the executions that succeeded are saved.

    Args:
        results_path: The path to the results file.
        rosenbrock: The problem configuration that succeeded.
        algorithm_configuration: The algorithm configuration.
    """
    with results_path.open() as results_file:
        data = json.load(results_file)

    algorithm_name = algorithm_configuration.name
    assert data[algorithm_name].keys() == {rosenbrock.name}
    assert len(data[algorithm_name][rosenbrock.name]) == 2


def test_results_saved_when_a_worker_raised(
    tmp_path, rosenbrock, optimization_algorithm_configuration
) -> None:
    """Check that the executions go on and are saved when a worker raised."""
    benchmarker, problem_configurations, algorithm_configurations, results_path = (
        create_failing_benchmarking(
            tmp_path, rosenbrock, optimization_algorithm_configuration
        )
    )
    benchmarker.execute(problem_configurations, algorithm_configurations)
    assert_succeeded_executions_saved(
        results_path, rosenbrock, optimization_algorithm_configuration
    )


def test_results_saved_when_a_worker_raised_and_errors_raised(
    tmp_path, rosenbrock, optimization_algorithm_configuration
) -> None:
    """Check that the executions are saved when errors are raised."""
    benchmarker, problem_configurations, algorithm_configurations, results_path = (
        create_failing_benchmarking(
            tmp_path, rosenbrock, optimization_algorithm_configuration
        )
    )
    with pytest.raises(
        BenchmarkingError, match=re.escape("1 of 3 executions raised an exception")
    ) as error_info:
        benchmarker.execute(
            problem_configurations, algorithm_configurations, raise_errors=True
        )

    assert len(error_info.value.exceptions) == 1
    assert_succeeded_executions_saved(
        results_path, rosenbrock, optimization_algorithm_configuration
    )


def create_unknown_algorithm_configuration(algorithm_name: str, name: str) -> mock.Mock:
    """Create the configuration of an algorithm unknown to GEMSEO.

    Args:
        algorithm_name: The name of the algorithm.
        name: The name of the configuration.

    Returns:
        The algorithm configuration.
    """
    algorithm_configuration = mock.Mock()
    algorithm_configuration.algorithm_name = algorithm_name
    algorithm_configuration.name = name
    return algorithm_configuration


def test_unavailable_algorithm_before_any_execution(
    tmp_path, rosenbrock, optimization_algorithm_configuration
) -> None:
    """Check that nothing is executed when the last algorithm is unavailable."""
    # The algorithm configurations are sorted by name:
    # the unknown algorithm comes after the available one.
    unknown_algorithm_configuration = create_unknown_algorithm_configuration(
        "Algorithm", "Z configuration"
    )
    algorithm_configurations = AlgorithmsConfigurations(
        optimization_algorithm_configuration, unknown_algorithm_configuration
    )
    assert list(algorithm_configurations)[-1] is unknown_algorithm_configuration
    with pytest.raises(
        ValueError, match=re.escape("The algorithm 'Algorithm' is not available.")
    ):
        Benchmarker(tmp_path, tmp_path / "results.json").execute(
            [rosenbrock], algorithm_configurations
        )

    assert not list(tmp_path.rglob("*.json"))


def test_all_unavailable_algorithms_reported(tmp_path, rosenbrock) -> None:
    """Check that all the unavailable algorithms are reported at once."""
    with pytest.raises(ValueError) as error_info:
        Benchmarker(tmp_path).execute(
            [rosenbrock],
            AlgorithmsConfigurations(
                create_unknown_algorithm_configuration("Algorithm", "Configuration"),
                create_unknown_algorithm_configuration("Other algorithm", "Other"),
            ),
        )

    assert str(error_info.value) == (
        "The algorithm 'Algorithm' is not available.\n"
        "The algorithm 'Other algorithm' is not available."
    )


def test_unavailable_algorithm_and_invalid_option_reported(
    tmp_path, rosenbrock, snapshot
) -> None:
    """Check that an unavailable algorithm and an invalid option are reported at once.

    Nothing is executed.
    """
    with assert_exception(ValueError, snapshot):
        Benchmarker(tmp_path).execute(
            [rosenbrock],
            AlgorithmsConfigurations(
                create_unknown_algorithm_configuration("Algorithm", "Configuration"),
                AlgorithmConfiguration(
                    L_BFGS_B_Settings(),
                    "L-BFGS-B",
                    instance_algorithm_options={
                        "max_iter": lambda configuration, index: "x"
                    },
                ),
            ),
        )

    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            (L_BFGS_B_Settings(),),
            (
                "The algorithm 'L_BFGS_B' is not adapted to the problem configuration "
                "'Power2': it does not handle equality constraints."
            ),
        ),
        (
            (L_BFGS_B_Settings(), TNC_Settings()),
            (
                "The algorithms 'L_BFGS_B', 'TNC' are not adapted "
                "to the problem configuration 'Power2':\n"
                "- 'L_BFGS_B': it does not handle equality constraints,\n"
                "- 'TNC': it does not handle equality constraints."
            ),
        ),
    ],
)
def test_unsuited_algorithm_before_any_execution(
    tmp_path, rosenbrock, settings, message
) -> None:
    """Check that nothing is executed when an algorithm is unsuited to a problem."""
    # Power2 has constraints, which neither L-BFGS-B nor TNC handles.
    power_2 = OptimizationProblemConfiguration("Power2", Power2)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        Benchmarker(tmp_path, tmp_path / "results.json").execute(
            [rosenbrock, power_2],
            AlgorithmsConfigurations(*(AlgorithmConfiguration(s) for s in settings)),
        )

    assert not list(tmp_path.rglob("*.json"))


@pytest.fixture
def failing_problem_creation(
    rosenbrock,
) -> Iterator[OptimizationProblemConfiguration]:
    """A problem configuration that can be created at instantiation, not afterwards.

    Yields:
        The problem configuration.
    """
    with mock.patch.object(
        OptimizationProblemConfiguration,
        "create_problem",
        side_effect=RuntimeError("Cannot create the problem."),
    ):
        yield rosenbrock


def assert_check_skipped_warning(caplog) -> None:
    """Check that the skipped check of the algorithms is logged as a warning.

    Args:
        caplog: The fixture to capture the logs.
    """
    assert any(
        record.levelno == logging.WARNING
        and "is not checked, as the problem cannot be created" in record.getMessage()
        and "Cannot create the problem." in record.getMessage()
        for record in caplog.records
    )


def test_problem_creation_failure_raised(
    tmp_path, failing_problem_creation, optimization_algorithm_configuration, caplog
) -> None:
    """Check that a failure to create a problem is raised as an execution failure."""
    with pytest.raises(BenchmarkingError) as error_info:
        Benchmarker(tmp_path).execute(
            [failing_problem_creation],
            AlgorithmsConfigurations(optimization_algorithm_configuration),
            use_threading=True,
            raise_errors=True,
        )

    assert all(
        isinstance(exception, RuntimeError)
        for exception in error_info.value.exceptions.values()
    )
    assert_check_skipped_warning(caplog)


def test_problem_creation_failure_not_raised(
    tmp_path, failing_problem_creation, optimization_algorithm_configuration, caplog
) -> None:
    """Check that a failure to create a problem is logged as an execution failure."""
    Benchmarker(tmp_path).execute(
        [failing_problem_creation],
        AlgorithmsConfigurations(optimization_algorithm_configuration),
        use_threading=True,
    )
    assert "Cannot create the problem." in caplog.text
    assert_check_skipped_warning(caplog)


def test_benchmarking_error_pickle() -> None:
    """Check that a benchmarking error survives a pickle round trip."""
    error = BenchmarkingError({"x": RuntimeError("boom")}, 2)
    unpickled = pickle.loads(pickle.dumps(error))  # ruff: ignore[suspicious-pickle-usage]
    assert str(unpickled) == str(error)
    assert unpickled.number_of_executions == 2
    assert list(unpickled.exceptions) == ["x"]
    assert str(unpickled.exceptions["x"]) == "boom"


@pytest.fixture
def unsuited_algorithm_histories(tmp_path) -> tuple[AlgorithmConfiguration, Path]:
    """The histories of an algorithm unsuited to a problem, saved in a results file.

    The problem configuration is `Power2`, which has constraints
    that the algorithm L-BFGS-B does not handle.

    Returns:
        The algorithm configuration and the path to the results file.
    """
    power_2 = OptimizationProblemConfiguration("Power2", Power2)
    algorithm_configuration = AlgorithmConfiguration(L_BFGS_B_Settings())
    benchmarker = Benchmarker(tmp_path)
    results = Results()
    for index in range(len(power_2.starting_points)):
        path = benchmarker.get_history_path(
            algorithm_configuration, power_2.name, index, True
        )
        path.touch()
        results.add_path(algorithm_configuration.name, power_2.name, path)

    results_path = tmp_path / "results.json"
    results.to_file(results_path)
    return algorithm_configuration, results_path


def test_unsuited_algorithm_with_existing_histories(
    tmp_path, unsuited_algorithm_histories
) -> None:
    """Check that an unsuited algorithm whose histories exist is not checked."""
    algorithm_configuration, results_path = unsuited_algorithm_histories
    power_2 = OptimizationProblemConfiguration("Power2", Power2)
    # The histories exist, so the algorithm is not checked against Power2,
    # and no execution is run: an execution would fail and raise an error.
    Benchmarker(tmp_path, results_path).execute(
        [power_2], AlgorithmsConfigurations(algorithm_configuration), raise_errors=True
    )

    # The histories created by the fixture are still empty.
    history_paths = [
        path for path in tmp_path.rglob("*") if path.is_file() and path != results_path
    ]
    assert history_paths
    assert all(path.stat().st_size == 0 for path in history_paths)


def test_unsuited_algorithm_with_overwritten_histories(
    tmp_path, unsuited_algorithm_histories
) -> None:
    """Check that an unsuited algorithm is checked if the histories are overwritten."""
    algorithm_configuration, results_path = unsuited_algorithm_histories
    power_2 = OptimizationProblemConfiguration("Power2", Power2)
    with pytest.raises(ValueError, match="not adapted"):
        Benchmarker(tmp_path, results_path).execute(
            [power_2],
            AlgorithmsConfigurations(algorithm_configuration),
            overwrite_histories=True,
        )


@pytest.mark.parametrize(
    ("number_of_executions", "first_line"),
    [
        (1, "1 of 1 execution raised an exception."),
        (
            2,
            (
                "1 of 2 executions raised an exception; "
                "the performance history of the other execution is saved."
            ),
        ),
        (
            3,
            (
                "1 of 3 executions raised an exception; "
                "the performance histories of the other executions are saved."
            ),
        ),
    ],
)
def test_benchmarking_error_message(number_of_executions, first_line) -> None:
    """Check the message of a benchmarking error, according to the executions."""
    error = BenchmarkingError({"x": RuntimeError("boom")}, number_of_executions)
    assert str(error) == f"{first_line}\n- x raised: RuntimeError: boom"
