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

"""Tests for the multidisciplinary analysis benchmarking worker."""

from pathlib import Path

import numpy
import pytest
from gemseo.caches.hdf5_cache import HDF5Cache
from gemseo.mda.factory import MDAFactory
from gemseo.mda.jacobi import MDAJacobi
from gemseo.utils.testing.pytest_conftest import tmp_wd  # noqa: F401
from gemseo.utils.timer import Timer

from gemseo_benchmark.benchmarker.mda_worker import MDAWorker
from tests.benchmarker.conftest import check_algorithm_availability
from tests.benchmarker.conftest import check_algorithm_factory
from tests.benchmarker.conftest import check_create_performance_history
from tests.benchmarker.conftest import check_execution


def test_algorithm_factory() -> None:
    """Check the algorithm factory."""
    check_algorithm_factory(MDAWorker, MDAFactory)


def test_check_algorithm_availability() -> None:
    """Check the algorithm availability checker."""
    check_algorithm_availability(MDAWorker)


@pytest.mark.parametrize("save_gemseo_log", [False, True])
@pytest.mark.parametrize("save_data", [False, True])
def test_execution(
    tmp_wd,  # noqa: F811
    mda_algorithm_configuration,
    mda_problem_configuration,
    save_gemseo_log,
    save_data,
) -> None:
    """Check the execution of the benchmarking worker."""
    check_execution(
        tmp_wd,
        mda_algorithm_configuration,
        mda_problem_configuration,
        save_gemseo_log,
        save_data,
        MDAWorker,
        "",
    )


@pytest.mark.parametrize("save_data", [False, True])
def test_get_problem(
    tmp_wd,  # noqa: F811
    mda_algorithm_configuration,
    mda_problem_configuration,
    save_data,
) -> None:
    """Check the problem getter."""
    hdf_file_path = Path("data.h5") if save_data else None
    mda = MDAWorker._get_problem(
        mda_algorithm_configuration,
        mda_problem_configuration,
        numpy.zeros(2),
        hdf_file_path,
    )[0]
    assert isinstance(mda, MDAJacobi)
    if save_data:
        assert isinstance(mda.cache, HDF5Cache)


def test_execute(
    mda_algorithm_configuration, mda_problem_configuration, enable_discipline_statistics
) -> None:
    """Check the execution of the benchmarking worker."""
    problem = mda_problem_configuration.create_problem(mda_algorithm_configuration)
    mda, (discipline1, discipline2) = problem
    assert mda.execution_statistics.n_executions == 0
    assert discipline1.execution_statistics.n_executions == 0
    assert discipline2.execution_statistics.n_executions == 0
    MDAWorker._execute(
        mda_algorithm_configuration,
        mda_problem_configuration,
        mda_problem_configuration.variable_space.get_lower_bounds(),
        problem,
    )
    assert mda.execution_statistics.n_executions == 1
    assert discipline1.execution_statistics.n_executions > 0
    assert discipline2.execution_statistics.n_executions > 0


def test_create_performance_history(
    mda_algorithm_configuration, mda_problem_configuration
) -> None:
    """Check the creation of the performance history."""
    problem = mda_problem_configuration.create_problem(mda_algorithm_configuration)
    mda = problem[0]
    with Timer() as timer:
        mda.execute()

    residual_history = mda.residual_history
    history_size = len(residual_history)
    check_create_performance_history(
        mda_algorithm_configuration,
        mda_problem_configuration,
        problem,
        MDAWorker,
        timer,
        residual_history,
        [0.0] * history_size,
        [0] * history_size,
    )
