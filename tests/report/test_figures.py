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

"""Tests for the figures of the report."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from matplotlib.testing.compare import compare_images

from gemseo_benchmark.algorithms.algorithms_configurations import (
    AlgorithmsConfigurations,
)
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.report._figures import Figures

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture
def problems_figures(tmp_path, algorithms_configurations, group, results) -> Figures:
    """The results figures for a group of problems."""
    return Figures(
        algorithms_configurations,
        group,
        results,
        tmp_path,
        0,
        0,
        {"SLSQP": {"color": "blue", "marker": "o"}},
    )


def test_plot_data_profiles(tmp_path, problems_figures):
    """Check the plotting of data profiles."""
    path = problems_figures.plot_data_profiles()
    assert path == tmp_path / "data_profile.png"
    assert path.is_file()


plot_options = pytest.mark.parametrize(
    (
        "plot_all_histories",
        "use_performance_log_scale",
        "plot_only_median",
        "use_time_log_scale",
        "use_abscissa_log_scale",
    ),
    [
        (False, False, False, False, False),
        (False, False, False, False, True),
        (False, False, False, True, False),
        (False, False, True, False, False),
        (False, True, False, False, False),
        (True, False, False, False, False),
    ],
)


@plot_options
def test_plot(
    tmp_path,
    problems_figures,
    plot_all_histories,
    use_performance_log_scale,
    plot_only_median,
    use_time_log_scale,
    use_abscissa_log_scale,
) -> None:
    """Check the plotting of the figures dedicated to each problem."""
    figures = problems_figures.plot(
        plot_all_histories,
        use_performance_log_scale,
        plot_only_median,
        use_time_log_scale,
        use_abscissa_log_scale,
    )[0]
    baseline_path = (
        Path(__file__).parent
        / "baseline_images"
        / "test_figures"
        / (
            "test_plot"
            f"-{plot_all_histories}"
            f"-{use_performance_log_scale}"
            f"-{plot_only_median}"
            f"-{use_time_log_scale}"
            f"-{use_abscissa_log_scale}"
        )
    )
    __check_problem_images(tmp_path, figures, "Problem_A", "SLSQP", baseline_path)
    __check_problem_images(tmp_path, figures, "Problem_B", "SLSQP", baseline_path)


def __check_problem_images(
    tmp_path: Path,
    paths: Mapping[str, Figures.ProblemFigurePaths],
    problem_name: str,
    algorithm_name: str,
    baseline_path: Path,
) -> None:
    """Check the plotting of the figures dedicated to a problem."""
    __check_problem_image(
        tmp_path,
        paths,
        problem_name,
        Figures._FigureFileName.PERFORMANCE_MEASURE,
        baseline_path,
    )
    __check_problem_image(
        tmp_path,
        paths,
        problem_name,
        Figures._FigureFileName.PERFORMANCE_MEASURE_FOCUS,
        baseline_path,
    )
    __check_problem_image(
        tmp_path,
        paths,
        problem_name,
        Figures._FigureFileName.INFEASIBILITY_MEASURE,
        baseline_path,
    )
    __check_problem_image(
        tmp_path,
        paths,
        problem_name,
        Figures._FigureFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS,
        baseline_path,
    )
    __check_problem_image(
        tmp_path,
        paths,
        problem_name,
        Figures._FigureFileName.EXECUTION_TIME,
        baseline_path,
    )
    __check_algorithm_image(
        tmp_path,
        paths,
        problem_name,
        algorithm_name,
        Figures._FigureFileName.PERFORMANCE_MEASURE,
        baseline_path,
    )
    __check_algorithm_image(
        tmp_path,
        paths,
        problem_name,
        algorithm_name,
        Figures._FigureFileName.PERFORMANCE_MEASURE_FOCUS,
        baseline_path,
    )
    __check_algorithm_image(
        tmp_path,
        paths,
        problem_name,
        algorithm_name,
        Figures._FigureFileName.INFEASIBILITY_MEASURE,
        baseline_path,
    )
    __check_algorithm_image(
        tmp_path,
        paths,
        problem_name,
        algorithm_name,
        Figures._FigureFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS,
        baseline_path,
    )


def __check_problem_image(
    tmp_path: Path,
    paths: Mapping[str, Figures.ProblemFigurePaths],
    problem_name: str,
    file_name: Figures._FigureFileName,
    baseline_path: Path,
) -> None:
    """Check the plotting of a figure dedicated to a problem."""
    __check_image(
        tmp_path,
        paths[problem_name.replace("_", " ")][file_name],
        tmp_path / problem_name / file_name.value,
        baseline_path,
    )


def __check_algorithm_image(
    tmp_path: Path,
    paths: Mapping[str, Figures.ProblemFigurePaths],
    problem_name: str,
    algorithm_name: str,
    file_name: Figures._FigureFileName,
    baseline_path: Path,
) -> None:
    """Check the plotting of a figure dedicated to an algorithm configuration."""
    __check_image(
        tmp_path,
        paths[problem_name.replace("_", " ")][algorithm_name][file_name],
        tmp_path / problem_name / algorithm_name / file_name.value,
        baseline_path,
    )


def __check_image(
    tmp_path: Path, generated_path: Path, expected_path: Path, baseline_path: Path
) -> None:
    """Check an image."""
    assert generated_path == expected_path
    assert (
        compare_images(
            baseline_path / generated_path.relative_to(tmp_path),
            generated_path,
            0,
            False,
        )
        is None
    )


def test_tables(tmp_path, problems_figures) -> None:
    """Check the tables dedicated to each problem."""
    tables = problems_figures.plot(False, False, False, False, False)[1]
    baseline_path = (
        Path(__file__).parent / "baseline_tables" / "test_figures" / "test_tables"
    )
    __check_problem_tables(tmp_path, tables, "Problem_A", "SLSQP", baseline_path)
    __check_problem_tables(tmp_path, tables, "Problem_B", "SLSQP", baseline_path)


def __check_problem_tables(
    tmp_path: Path,
    paths: Mapping[str, Figures.ProblemTablePaths],
    problem_name: str,
    algorithm_name: str,
    baseline_path: Path,
) -> None:
    """Check the tables dedicated to a problem."""
    __check_problem_table(
        tmp_path,
        paths,
        problem_name,
        Figures._TableFileName.PERFORMANCE_MEASURE,
        baseline_path,
    )
    __check_problem_table(
        tmp_path,
        paths,
        problem_name,
        Figures._TableFileName.INFEASIBILITY_MEASURE,
        baseline_path,
    )
    __check_problem_table(
        tmp_path,
        paths,
        problem_name,
        Figures._TableFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS,
        baseline_path,
    )
    __check_algorithm_table(
        tmp_path,
        paths,
        problem_name,
        algorithm_name,
        Figures._TableFileName.PERFORMANCE_MEASURE,
        baseline_path,
    )
    __check_algorithm_table(
        tmp_path,
        paths,
        problem_name,
        algorithm_name,
        Figures._TableFileName.INFEASIBILITY_MEASURE,
        baseline_path,
    )
    __check_algorithm_table(
        tmp_path,
        paths,
        problem_name,
        algorithm_name,
        Figures._TableFileName.NUMBER_OF_UNSATISFIED_CONSTRAINTS,
        baseline_path,
    )


def __check_problem_table(
    tmp_path: Path,
    paths: Mapping[str, Figures.ProblemTablePaths],
    problem_name: str,
    file_name: Figures._TableFileName,
    baseline_path: Path,
) -> None:
    """Check a table dedicated to a problem."""
    __check_table(
        tmp_path,
        paths[problem_name.replace("_", " ")][file_name],
        tmp_path / problem_name / file_name.value,
        baseline_path,
    )


def __check_algorithm_table(
    tmp_path: Path,
    paths: Mapping[str, Figures.ProblemTablePaths],
    problem_name: str,
    algorithm_name: str,
    file_name: Figures._TableFileName,
    baseline_path: Path,
) -> None:
    """Check a table dedicated to an algorithm configuration."""
    __check_table(
        tmp_path,
        paths[problem_name.replace("_", " ")][algorithm_name][file_name],
        tmp_path / problem_name / algorithm_name / file_name.value,
        baseline_path,
    )


def __check_table(
    tmp_path: Path, generated_path: Path, expected_path: Path, baseline_path: Path
) -> None:
    """Check a table."""
    assert generated_path == expected_path
    assert (
        generated_path.open().readlines()
        == (baseline_path / generated_path.relative_to(tmp_path)).open().readlines()
    )


def test_no_data_profiles_duplicates(
    tmp_path,
    algorithms_configurations,
    problem_a,
    results,
) -> None:
    """Check that data profiles are not computed twice for a single problem."""
    group = mock.MagicMock()
    group.__iter__.return_value = [problem_a]
    group.__len__.return_value = 1

    def side_effect(*args, **kwargs):
        shutil.copyfile(
            str(Path(__file__).parent.parent / "data_profile.png"),
            str(tmp_path / "data_profile.png"),
        )

    group.compute_data_profile = mock.Mock(side_effect=side_effect)
    figures = Figures(
        algorithms_configurations,
        group,
        results,
        tmp_path,
        0,
        0,
        {"SLSQP": {"color": "blue", "marker": "o"}},
    )
    group_call_count = group.compute_data_profile.call_count
    problem_call_count = problem_a.compute_data_profile.call_count
    figures.plot_data_profiles()
    figures.plot(False, False, False, False, False)
    assert group.compute_data_profile.call_count == group_call_count + 1
    assert problem_a.compute_data_profile.call_count == problem_call_count


@pytest.fixture
def figures_unequal_representation(
    tmp_path, unknown_algorithms_configurations, group, results
) -> Figures:
    """The results figures for unequally represented algorithm configurations."""
    paths = results.get_paths.return_value
    results.get_paths = mock.MagicMock(side_effect=[paths, paths[:1]] * 2)
    return Figures(
        unknown_algorithms_configurations,
        group,
        results,
        tmp_path,
        0,
        0,
        {"SLSQP": {"color": "blue", "marker": "o"}},
    )


def test_unequal_representation(
    figures_unequal_representation,
    problem_a,
    algorithm_configuration,
    unknown_algorithm_configuration,
) -> None:
    """Check the handling of unequally represented algorithm configurations."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The number of performance histories varies for '{problem_a.name}': "
            f"2 for '{algorithm_configuration.name}' "
            f"and 1 for '{unknown_algorithm_configuration.name}'."
        ),
    ):
        figures_unequal_representation.plot(False, False, False, False, False)


def test_bypass_unequal_representation(
    figures_unequal_representation,
    problem_a,
    problem_b,
    algorithm_configuration,
    unknown_algorithm_configuration,
    caplog,
) -> None:
    """Check the handling of unequally represented algorithm configurations."""
    with caplog.at_level(logging.WARNING):
        figures_unequal_representation.plot(
            False, False, False, False, False, bypass_unequal_representation=True
        )

    assert caplog.record_tuples == [
        (
            "gemseo_benchmark.report._figures",
            logging.WARNING,
            (
                f"The number of performance histories varies for '{problem.name}': "
                f"2 for '{algorithm_configuration.name}' "
                f"and 1 for '{unknown_algorithm_configuration.name}'."
            ),
        )
        for problem in [problem_a, problem_b]
    ]


@pytest.mark.parametrize("problem", ["problem_a", "problem_b"])
@plot_options
def test_plot_infeasible(
    algorithm_configuration,
    problem,
    tmp_path,
    plot_all_histories,
    use_performance_log_scale,
    plot_only_median,
    use_time_log_scale,
    use_abscissa_log_scale,
    request,
) -> None:
    """Check the plotting and tabulating of infeasible performance measures."""
    problem = request.getfixturevalue(problem)
    infeasible_results = mock.Mock()
    infeasible_results.get_paths = mock.Mock(
        return_value=[Path(__file__).parent.parent / "unfeasible_history.json"]
    )
    figures, tables = Figures(
        AlgorithmsConfigurations(algorithm_configuration),
        ProblemsGroup("Singleton", [problem]),
        infeasible_results,
        tmp_path,
        0,
        0,
        {"SLSQP": {"color": "blue", "marker": "o"}},
    ).plot(
        plot_all_histories,
        use_performance_log_scale,
        plot_only_median,
        use_time_log_scale,
        use_abscissa_log_scale,
    )
    baseline_path = (
        Path(__file__).parent
        / "baseline_images"
        / "test_figures"
        / (
            "test_plot_infeasible"
            f"-{plot_all_histories}"
            f"-{use_performance_log_scale}"
            f"-{plot_only_median}"
            f"-{use_time_log_scale}"
            f"-{use_abscissa_log_scale}"
        )
    )
    algorithm_name = algorithm_configuration.algorithm_name
    problem_name = problem.name.replace(" ", "_")
    __check_problem_images(
        tmp_path, figures, problem_name, algorithm_name, baseline_path
    )
    __check_problem_tables(
        tmp_path, tables, problem_name, algorithm_name, baseline_path
    )
