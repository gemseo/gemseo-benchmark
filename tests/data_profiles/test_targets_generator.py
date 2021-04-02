from pytest import raises

from data_profiles.targets_generator import TargetsGenerator


def test_add_inconsistent_histories():
    """Check the addition of inconsistent performance histories to the targets
    generator"""
    generator = TargetsGenerator()
    with raises(ValueError):
        generator.add_history([3.0, 2.0], [1.0])
    with raises(ValueError):
        generator.add_history([3.0, 2.0], feasibility_history=[False])


def test_negative_infeasibility_measures():
    """Check the addition of a performance history with negative infeasibility
    measures"""
    generator = TargetsGenerator()
    with raises(ValueError):
        generator.add_history([3.0, 2.0], [1.0, -1.0])
