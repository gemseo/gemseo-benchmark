from pathlib import Path

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


def test_save_to_file(tmp_path):
    """Check the writing of a performance history into a file."""
    history = PerformanceHistory([-2.0, -3.0], [1.0, 0.0])
    file_path = tmp_path / "history.json"
    history.save_to_file(file_path)
    with open(file_path) as the_file:
        contents = the_file.read()
    reference_path = Path(__file__).parent / "reference_history.json"
    with open(reference_path) as reference_file:
        reference = reference_file.read()
    assert contents == reference


def test_load_from_file():
    """Check the initialization of a perfomance history from a file."""
    reference_path = Path(__file__).parent / "reference_history.json"
    history = PerformanceHistory.load_from_file(reference_path)
    assert history.to_list() == [(-2.0, 1.0), (-3.0, 0.0)]
