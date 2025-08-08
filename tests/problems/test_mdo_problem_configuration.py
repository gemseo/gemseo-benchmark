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
"""Tests for MDO problem configurations."""

from unittest import mock

from matplotlib.testing.decorators import image_comparison

from gemseo_benchmark.benchmarker.mdo_worker import MDOWorker
from gemseo_benchmark.problems.mdo_problem_configuration import MDOProblemConfiguration
from tests.conftest import mdo_create_problem
from tests.problems.conftest import check_data_profiles_computation
from tests.problems.conftest import check_data_profiles_computation_with_max_eval_number
from tests.problems.conftest import check_default_starting_point
from tests.problems.conftest import check_description
from tests.problems.conftest import check_inconsistent_starting_points
from tests.problems.conftest import check_no_starting_point
from tests.problems.conftest import check_set_starting_points_as_non_2d_array
from tests.problems.conftest import check_set_starting_points_as_non_iterable
from tests.problems.conftest import check_set_starting_points_with_wrong_dimension
from tests.problems.conftest import check_starting_point_loading
from tests.problems.conftest import check_starting_points_default_generation
from tests.problems.conftest import check_starting_points_generation
from tests.problems.conftest import check_starting_points_saving
from tests.problems.conftest import check_undefined_target_values
from tests.problems.conftest import check_variable_space
from tests.problems.conftest import check_worker_type
from tests.problems.conftest import test_compute_data_profiles_parametrize
from tests.problems.conftest import test_description_parametrize


def test_undefined_starting_points() -> None:
    """Check the absence of starting points."""
    variable_space = mock.Mock()
    variable_space.has_current_value = False
    check_no_starting_point(
        MDOProblemConfiguration(
            "Linear MDO", mdo_create_problem, variable_space, True, 0
        )
    )


def test_default_starting_point(multidisciplinary_variable_space) -> None:
    """Check that the default starting point is properly set."""
    check_default_starting_point(
        MDOProblemConfiguration(
            "Linear MDO", mdo_create_problem, multidisciplinary_variable_space, True, 0
        ),
        multidisciplinary_variable_space,
    )


def test_inconsistent_starting_points(multidisciplinary_variable_space) -> None:
    """Check initialization with starting points of inadequate size."""
    check_inconsistent_starting_points(
        MDOProblemConfiguration,
        mdo_create_problem,
        multidisciplinary_variable_space,
        True,
        0,
    )


def test_undefined_target_values(multidisciplinary_variable_space) -> None:
    """Check undefined target values."""
    check_undefined_target_values(
        MDOProblemConfiguration(
            "Linear MDO", mdo_create_problem, multidisciplinary_variable_space, True, 0
        )
    )


def test_generate_default_starting_points(multidisciplinary_variable_space) -> None:
    """Check the generation of the default number of starting points."""
    check_starting_points_default_generation(
        MDOProblemConfiguration,
        mdo_create_problem,
        multidisciplinary_variable_space,
        True,
        0,
    )


def test_generate_starting_points(multidisciplinary_variable_space) -> None:
    """Check the generation of starting points."""
    check_starting_points_generation(
        MDOProblemConfiguration,
        mdo_create_problem,
        multidisciplinary_variable_space,
        True,
        0,
    )


def test_starting_points_non_2d_array(mdo_problem_configuration) -> None:
    """Check the setting of starting points as a non 2-dimensional NumPy array."""
    check_set_starting_points_as_non_2d_array(mdo_problem_configuration)


def test_starting_points_non_iterable(mdo_problem_configuration) -> None:
    """Check the setting of starting points as a non-iterable object."""
    check_set_starting_points_as_non_iterable(mdo_problem_configuration)


def test_starting_points_wrong_dimension(mdo_problem_configuration) -> None:
    """Check the setting of starting points of the wrong dimension."""
    check_set_starting_points_with_wrong_dimension(mdo_problem_configuration)


@test_description_parametrize
def test_description(
    mdo_problem_configuration,
    multidisciplinary_variable_space,
    input_description,
    actual_description,
) -> None:
    """Check the description."""
    check_description(
        MDOProblemConfiguration,
        mdo_problem_configuration,
        input_description,
        actual_description,
        multidisciplinary_variable_space,
        True,
        0,
    )


def test_save_starting_points(tmp_path, multidisciplinary_variable_space) -> None:
    """Check the saving of starting points."""
    check_starting_points_saving(
        tmp_path,
        MDOProblemConfiguration,
        mdo_create_problem,
        multidisciplinary_variable_space,
        True,
        0,
    )


def test_load_starting_points(tmp_path, multidisciplinary_variable_space) -> None:
    """Check the loading of starting points."""
    check_starting_point_loading(
        tmp_path,
        MDOProblemConfiguration,
        mdo_create_problem,
        multidisciplinary_variable_space,
        True,
        0,
    )


def test_dimension(mdo_problem_configuration) -> None:
    """Check the problem dimension."""
    assert mdo_problem_configuration.dimension == 2


@test_compute_data_profiles_parametrize
@image_comparison(None, ["png"])
def test_compute_data_profile(
    baseline_images,
    multidisciplinary_variable_space,
    target_values,
    algorithms_configurations,
    results,
    use_abscissa_log_scale,
) -> None:
    """Check the computation of data profiles."""
    check_data_profiles_computation(
        MDOProblemConfiguration,
        mdo_create_problem,
        target_values,
        algorithms_configurations,
        results,
        use_abscissa_log_scale,
        multidisciplinary_variable_space,
        True,
        0,
    )


@image_comparison(
    baseline_images=["data_profiles_max_eval_number"],
    remove_text=True,
    extensions=["png"],
)
def test_compute_data_profile_max_eval_number(
    multidisciplinary_variable_space,
    target_values,
    algorithms_configurations,
    results,
) -> None:
    """Check the computation of data profiles when the evaluations number is limited."""
    check_data_profiles_computation_with_max_eval_number(
        MDOProblemConfiguration,
        mdo_create_problem,
        target_values,
        algorithms_configurations,
        results,
        multidisciplinary_variable_space,
        True,
        0,
    )


def test_minimize_performance_measure(mdo_problem_configuration) -> None:
    """Check the minimization flag."""
    assert mdo_problem_configuration.minimize_performance_measure


def test_variable_space(
    mdo_problem_configuration, multidisciplinary_variable_space
) -> None:
    """Check the variable space."""
    check_variable_space(mdo_problem_configuration, multidisciplinary_variable_space)


def test_worker(mdo_problem_configuration) -> None:
    """Check the type of benchmarking worker."""
    check_worker_type(mdo_problem_configuration, MDOWorker)
