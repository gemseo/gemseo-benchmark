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
"""Tests for the benchmarking worker."""

from __future__ import annotations

from gemseo_benchmark.benchmarker.worker import Worker


def test_call(algorithm_configuration, rosenbrock) -> None:
    """Check the call to the benchmarking worker."""
    _, problem_instance_index, database, history = Worker()((
        algorithm_configuration,
        rosenbrock,
        rosenbrock.create_problem(),
        0,
        None,
    ))
    assert problem_instance_index == 0
    assert len(database) > 0
    assert len(history) > 0
    assert history.algorithm_configuration == algorithm_configuration
    assert history.doe_size == 1
    assert history.total_time > 0


def test_gemseo_log_file(algorithm_configuration, rosenbrock, tmp_path) -> None:
    """Check logging GEMSEO to file."""
    path = tmp_path / "gemseo.log"
    assert not path.is_file()
    Worker()((
        algorithm_configuration,
        rosenbrock,
        rosenbrock.create_problem(),
        0,
        path,
    ))
    with path.open("r") as file:
        assert """Optimization problem:
   minimize rosen(x) = sum( 100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2 )
   with respect to x
   over the design space:
      +------+-------------+-------+-------------+-------+
      | Name | Lower bound | Value | Upper bound | Type  |
      +------+-------------+-------+-------------+-------+
      | x[0] |      -2     |   0   |      2      | float |
      | x[1] |      -2     |   0   |      2      | float |
      +------+-------------+-------+-------------+-------+
Solving optimization problem with algorithm SLSQP:""" in file.read()
