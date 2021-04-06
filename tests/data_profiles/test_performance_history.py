from numpy import inf
from pytest import raises

from data_profiles.performance_history import PerformanceHistory


def test_invalid_init_lengths():
    """Check the initialization of a performance history with lists of inconsistent
    lengths"""
    with raises(ValueError):
        PerformanceHistory([3.0, 2.0], [1.0])
    with raises(ValueError):
        PerformanceHistory([3.0, 2.0], feasibility=[False])


def test_negative_infeasibility_measures():
    """Check the initialization of a performance history with negative infeasibility
    measures"""
    with raises(ValueError):
        PerformanceHistory([3.0, 2.0], [1.0, -1.0])


def test_length():
    """Check the length of a performance history"""
    history_1 = PerformanceHistory([3.0, 2.0])
    assert len(history_1) == 2
    history_2 = PerformanceHistory([3.0, 2.0], [1.0, 0.0])
    assert len(history_2) == 2
    history_3 = PerformanceHistory([3.0, 2.0], feasibility=[False, True])
    assert len(history_3) == 2


def test_to_list():
    """Check the conversion of a performance history to a list"""
    history_1 = PerformanceHistory([3.0, 2.0])
    assert history_1.to_list() == [(3.0, 0.0), (2.0, 0.0)]
    history_2 = PerformanceHistory([3.0, 2.0], [1.0, 0.0])
    assert history_2.to_list() == [(3.0, 1.0), (2.0, 0.0)]
    history_3 = PerformanceHistory([3.0, 2.0], feasibility=[False, True])
    assert history_3.to_list() == [(3.0, inf), (2.0, 0.0)]


def test_iter():
    """Check the iteration over a performance history"""
    history = PerformanceHistory([3.0, 2.0], [1.0, 0.0])
    assert list(iter(history)) == [(3.0, 1.0), (2.0, 0.0)]
    history = PerformanceHistory([3.0, 2.0], feasibility=[False, True])
    assert list(iter(history)) == [(3.0, inf), (2.0, 0.0)]


def test_cumulated_min():
    """Check the cumulated minimum of a performance history"""
    history = PerformanceHistory(
        [0.0, -3.0, -1.0, 0.0, 1.0, -1.0],
        [2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    )
    acc_min = history.cumulated_min_history()
    assert acc_min.to_list() == [
        (0.0, 2.0), (0.0, 2.0), (-1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)
    ]
