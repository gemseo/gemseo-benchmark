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
"""Benchmarking of algorithms."""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING
from typing import Union

import matplotlib

if TYPE_CHECKING:
    from collections.abc import Iterator

MarkeveryType = Union[
    None, int, tuple[int, int], slice, list[int], float, tuple[float, float], list[bool]
]
# The colors cycle for the plots
COLORS_CYCLE = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
# The markers for the plots
MARKERS = ("o", "s", "D", "v", "^", "<", ">", "X", "H", "p")


def get_markers_cycle() -> Iterator:
    """Return the markers cycle for the plots.

    Returns:
        The markers cycle.
    """
    return itertools.cycle(MARKERS)


def join_substrings(string: str) -> str:
    """Join sub-strings with underscores.

    Args:
        string: The string.

    Returns:
        The joined sub-strings.
    """
    return re.sub(r"\s+", "_", string)
