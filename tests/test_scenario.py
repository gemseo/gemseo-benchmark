"""Tests for the benchmarking scenario."""
import pytest

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo_benchmark.algorithms.algorithm_configuration import AlgorithmConfiguration
from gemseo_benchmark.algorithms.algorithms_configurations import \
    AlgorithmsConfigurations
from gemseo_benchmark.problems.problems_group import ProblemsGroup
from gemseo_benchmark.scenario import Scenario


def test_inexistent_outputs_path(algorithms_configurations):
    """Check the handling of a nonexistent path of the outputs."""
    outputs_path = "/not/a/path/"
    with pytest.raises(
            NotADirectoryError,
            match=f"The path to the outputs directory does not exist: {outputs_path}."
    ):
        Scenario(algorithms_configurations, outputs_path)


@pytest.mark.parametrize("save_databases", [False, True])
@pytest.mark.parametrize("save_pseven_logs", [False, True])
def test_execute(
    algorithms_configurations, tmp_path, rosenbrock, save_databases, save_pseven_logs
):
    """Check the execution of a benchmarking scenario."""
    Scenario(algorithms_configurations, tmp_path).execute(
        [ProblemsGroup("Rosenbrock", [rosenbrock])],
        save_databases=save_databases,
        save_pseven_logs=save_pseven_logs
    )
    assert (tmp_path / "histories").is_dir()
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "report").is_dir()
    assert (tmp_path / "databases").is_dir() == save_databases
    assert not (tmp_path / "pseven_logs").is_dir()


@pytest.mark.skipif(
    not OptimizersFactory().is_available("PSEVEN"), reason="pSeven is not available."
)
@pytest.mark.parametrize("save_pseven_logs", [False, True])
def test_execute(algorithms_configurations, tmp_path, rosenbrock, save_pseven_logs):
    """Check the execution of a benchmarking scenario."""
    Scenario(
        AlgorithmsConfigurations(AlgorithmConfiguration("PSEVEN")), tmp_path
    ).execute(
        [ProblemsGroup("Rosenbrock", [rosenbrock])],
        save_pseven_logs=save_pseven_logs
    )
    assert (tmp_path / "pseven_logs").is_dir() == save_pseven_logs
