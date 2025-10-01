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
"""Problem configuration for multidisciplinary analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from gemseo.core.discipline.discipline import Discipline
from gemseo.mda.base_mda_solver import BaseMDASolver
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

from gemseo_benchmark.benchmarker.mda_worker import MDAWorker
from gemseo_benchmark.problems.base_problem_configuration import (
    BaseProblemConfiguration,
)
from gemseo_benchmark.report.axis_data import DisciplineData

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.base_doe_library import DriverLibraryOptionType

    from gemseo_benchmark.algorithms.algorithm_configuration import (
        AlgorithmConfiguration,
    )
    from gemseo_benchmark.data_profiles.target_values import TargetValues
    from gemseo_benchmark.problems.base_benchmarking_problem import (
        InputStartingPointsType,
    )

MDAProblemType = tuple[BaseMDASolver, Sequence[Discipline]]


class MDAProblemConfiguration(BaseProblemConfiguration):
    """Problem configuration for multidisciplinary analysis."""

    abscissa_data_type: Final[type[DisciplineData]] = DisciplineData
    performance_measure_label: ClassVar[str] = "Best residual norm"
    worker: ClassVar[type[MDAWorker]] = MDAWorker

    def __init__(  # noqa: D107
        self,
        name: str,
        create_problem: Callable[[AlgorithmConfiguration], MDAProblemType],
        variable_space: DesignSpace,
        target_values: TargetValues | None = None,
        starting_points: InputStartingPointsType = (),
        doe_algo_name: str = "",
        doe_size: int | None = None,
        doe_options: Mapping[str, DriverLibraryOptionType] = READ_ONLY_EMPTY_DICT,
        description: str = "No description available.",
    ) -> None:
        super().__init__(
            name,
            create_problem,
            target_values,
            starting_points,
            variable_space,
            doe_algo_name,
            doe_size,
            doe_options,
            description,
            0,
            0,
        )

    @property
    def minimize_performance_measure(self) -> bool:
        """Whether the performance measure is to be minimized."""
        return True
