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
"""The metrics of a benchmarking execution."""

from __future__ import annotations

import datetime
from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo import configure
from gemseo.mda.base_mda_solver import BaseMDASolver
from gemseo.typing import RealArray
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.core.discipline.discipline import Discipline
    from gemseo.utils.timer import Timer

MetricsDataType = RealArray | BaseMDASolver


class BaseMetrics(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Execution metrics."""

    _metrics: list[datetime.timedelta] | list[int]
    """The execution metrics."""

    def __init__(self) -> None:
        self._metrics = []

    @abstractmethod
    def add_metrics(self, data: MetricsDataType) -> None:
        """Add execution metrics.

        Args:
            data: The design variables of the optimization problem,
                or the multidisciplinary analysis solver.
        """

    @abstractmethod
    def get_metrics(self) -> list[datetime.timedelta] | list[int]:
        """Return the execution metrics.

        Returns:
            The execution metrics.
        """


class ElapsedTime(BaseMetrics):
    """The elapsed time of an execution."""

    def add_metrics(self, data: MetricsDataType) -> None:
        self._metrics.append(datetime.datetime.now())

    def get_metrics(self, timer: Timer) -> list[datetime.timedelta]:
        """
        Args:
            timer: The timer of the execution.
        """  # noqa: D205, D212
        start_datetime = timer.entering_timestamp
        return [end_datetime - start_datetime for end_datetime in self._metrics]


class DisciplineExecutions(BaseMetrics):
    """The number of discipline executions of an execution."""

    def __init__(self, disciplines: Iterable[Discipline]) -> None:
        """
        Args:
            disciplines: The disciplines.
        """  # noqa: D205, D212
        configure(enable_discipline_statistics=True)
        super().__init__()
        self.__disciplines = disciplines

    def add_metrics(self, data: MetricsDataType) -> None:
        self._metrics.append(
            sum(
                discipline.execution_statistics.n_executions
                for discipline in self.__disciplines
            )
        )

    def get_metrics(self) -> list[int]:
        return self._metrics
