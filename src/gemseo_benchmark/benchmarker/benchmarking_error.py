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

"""The error raised when executions of a benchmarking raised exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Mapping


class BenchmarkingError(Exception):
    """The error raised when executions of a benchmarking raised exceptions.

    The other executions, if any, have run to the end
    and their performance histories are saved.
    The exception raised by the first execution that failed,
    in the order of submission, is the cause of this error.
    """

    exceptions: Mapping[str, BaseException]
    """The exceptions raised by the executions, bound to the description of each."""

    number_of_executions: int
    """The number of executions."""

    def __init__(
        self, exceptions: Mapping[str, BaseException], number_of_executions: int
    ) -> None:
        """
        Args:
            exceptions: The exceptions raised by the executions,
                bound to the description of each.
            number_of_executions: The number of executions.
        """  # ruff: ignore[missing-blank-line-after-summary, multi-line-summary-first-line, missing-terminal-punctuation]
        self.exceptions = exceptions
        self.number_of_executions = number_of_executions
        message = (
            f"{len(exceptions)} of {number_of_executions} "
            f"{'execution' if number_of_executions == 1 else 'executions'} "
            "raised an exception"
        )
        number_of_others = number_of_executions - len(exceptions)
        if number_of_others == 1:
            message += "; the performance history of the other execution is saved"
        elif number_of_others > 1:
            message += "; the performance histories of the other executions are saved"

        lines = [f"{message}."]
        lines.extend(
            f"- {description} raised: {type(exception).__name__}: {exception}"
            for description, exception in exceptions.items()
        )
        super().__init__("\n".join(lines))

    def __reduce__(self) -> tuple[type[BenchmarkingError], tuple[Any, ...]]:
        """Return the arguments to rebuild the error when unpickling.

        Returns:
            The class of the error and the arguments of its constructor.
        """
        return type(self), (self.exceptions, self.number_of_executions)
