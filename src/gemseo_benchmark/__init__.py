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
from collections.abc import Mapping
from typing import TYPE_CHECKING

import matplotlib

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

MarkeveryType = (
    int | tuple[int] | slice | list[int] | float | tuple[float] | list[bool] | None
)
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


ConfigurationPlotOptions = Mapping[str, str]


__DEFAULT_OPTIONS = {"color": lambda: COLORS_CYCLE, "marker": get_markers_cycle}


def _get_configuration_plot_options(
    options: Mapping[str, ConfigurationPlotOptions],
    names: Iterable[str],
) -> dict[str, str]:
    """Return the plot options of algorithm configurations.

    Args:
        options: The plot options of the algorithm configurations.
        names: The names of the algorithm configurations.

    Returns:
        The plot options of each algorithm configuration.
    """
    options = options.copy()
    for configuration_name in names:
        if configuration_name in options:
            options[configuration_name]["label"] = configuration_name
        else:
            options[configuration_name] = {"label": configuration_name}

    for option_name in __DEFAULT_OPTIONS:
        for configuration_name, default_value in zip(
            (
                configuration_name
                for configuration_name in names
                if option_name not in options[configuration_name]
            ),
            __DEFAULT_OPTIONS[option_name](),
            strict=False,
        ):
            options[configuration_name][option_name] = default_value

    return options
