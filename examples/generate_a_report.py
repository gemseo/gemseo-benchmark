from pathlib import Path

from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import zeros

from problems.problem import BenchmarkingProblem
from problems.problems_group import ProblemsGroup
from report.report import Report

reference_algorithms = {"SLSQP": dict()}
comparison_algorithms = {"NLOPT_SLSQP": dict()}
a_problem = BenchmarkingProblem("Rosenbrock 2D", Rosenbrock, [zeros(2)])
a_problem.generate_targets(20, reference_algorithms)
problems_groups = [ProblemsGroup("A group", [a_problem], "A description")]

report_path = Path(__file__).parent / "report"
report = Report(report_path, comparison_algorithms, problems_groups, dict())
report.generate_report_sources()
