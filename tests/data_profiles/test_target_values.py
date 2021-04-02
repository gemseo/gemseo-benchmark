from data_profiles.performance_history import PerformanceHistory
from data_profiles.target_values import TargetValues


def test_count_targets_hist():
    """Check the counting for hit targets"""
    targets = TargetValues([-2.0, 1.0, -1.0], [1.0, 0.0, 0.0])
    history = PerformanceHistory(
        [0.0, -3.0, -1.0, 0.0, 1.0, -1.0], [2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    )
    assert targets.count_targets_hits(history) == [0, 0, 1, 2, 2, 3]
