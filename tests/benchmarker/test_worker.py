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

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo_benchmark.benchmarker.worker import Worker


def test_call(algorithm_configuration, rosenbrock):
    """Check the call to the benchmarking worker."""
    _, problem_instance_index, database, history = Worker(OptimizersFactory())(
        (algorithm_configuration, rosenbrock, rosenbrock.creator(), 0)
    )
    assert problem_instance_index == 0
    assert len(database) > 0
    assert len(history) > 0
    assert history.algorithm_configuration == algorithm_configuration
    assert history.doe_size == 1
    assert history.total_time > 0
