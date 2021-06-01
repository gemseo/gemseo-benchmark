# -*- coding: utf-8 -*-
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

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the targets generator."""
from pytest import raises

from data_profiles.targets_generator import TargetsGenerator


def test_add_inconsistent_histories():
    """Check the addition of inconsistent performance histories."""
    generator = TargetsGenerator()
    with raises(ValueError):
        generator.add_history([3.0, 2.0], [1.0])
    with raises(ValueError):
        generator.add_history([3.0, 2.0], feasibility_statuses=[False])


def test_negative_infeasibility_measures():
    """Check the addition of a history with negative infeasibility measures."""
    generator = TargetsGenerator()
    with raises(ValueError):
        generator.add_history([3.0, 2.0], [1.0, -1.0])
