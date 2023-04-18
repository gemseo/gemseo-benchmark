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
