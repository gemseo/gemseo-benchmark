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

import shutil
from pathlib import Path
from unittest import mock

import pytest

from gemseo_benchmark.report._figures import Figures


@pytest.fixture
def problems_figures(tmp_path, algorithms_configurations, group, results) -> Figures:
    """The results figures for a group of problems."""
    return Figures(
        algorithms_configurations,
        group,
        results,
        tmp_path,
        0,
        None,
        {"SLSQP": {"color": "blue", "marker": "o"}},
    )


def test_plot_data_profiles(tmp_path, problems_figures):
    """Check the plotting of data profiles."""
    path = problems_figures.plot_data_profiles()
    assert path == tmp_path / "data_profile.png"
    assert path.is_file()


@pytest.mark.parametrize("plot_all_histories", [False, True])
@pytest.mark.parametrize("use_performance_log_scale", [False, True])
@pytest.mark.parametrize("plot_only_median", [False, True])
@pytest.mark.parametrize("use_time_log_scale", [False, True])
def test_plot(
    tmp_path,
    problems_figures,
    plot_all_histories,
    use_performance_log_scale,
    plot_only_median,
    use_time_log_scale,
) -> None:
    """Check the plotting of the figures dedicated to each problem."""
    figures = problems_figures.plot(
        plot_all_histories,
        use_performance_log_scale,
        plot_only_median,
        use_time_log_scale,
    )
    __check_problem_paths(tmp_path, figures["Problem A"], "Problem_A")
    __check_problem_paths(tmp_path, figures["Problem B"], "Problem_B")


def __check_problem_paths(tmp_path, paths, name) -> None:
    __check_path(tmp_path, paths, name, Figures._FileName.DATA_PROFILE)
    __check_path(tmp_path, paths, name, Figures._FileName.PERFORMANCE_MEASURE)
    __check_path(tmp_path, paths, name, Figures._FileName.PERFORMANCE_MEASURE_FOCUS)


def __check_path(tmp_path, paths, problem_name, file_name) -> None:
    path = paths[file_name]
    assert path == tmp_path / problem_name / file_name.value
    assert path.is_file()


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
        None,
        {"SLSQP": {"color": "blue", "marker": "o"}},
    )
    group_call_count = group.compute_data_profile.call_count
    problem_call_count = problem_a.compute_data_profile.call_count
    figures.plot_data_profiles()
    figures.plot(False, False, False, False)
    assert group.compute_data_profile.call_count == group_call_count + 1
    assert problem_a.compute_data_profile.call_count == problem_call_count
