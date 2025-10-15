<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Changelog titles are:
- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.
-->

# Changelog

All notable changes of this project will be documented here.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0)
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# Version 4.0.1 (October 2025)

## Added

- Support for Python 3.13.

## Removed

- Support for Python 3.9.

# Version 4.0.0 (August 2025)

## Added

### Problems

- Multidisciplinary analysis problem configurations
  can now be implemented with ``MDAProblemConfiguration``.
- Multidisciplinary optimization problem configurations
  can now be implemented with ``MDOProblemConfiguration``.

#### Report

- The plot options (ex: color, marker) of each algorithm configuration
  can now be customized at the execution of a ``Scenario``
  thanks to the new argument ``plot_settings``.
- The user can now request that ``Scenario.execute`` or ``Report.generate``
  plot only the median of the performance measure rather than its whole range
  thanks to the new boolean argument ``plot_only_median``.
- On the page dedicated to the benchmarking problems,
  the infeasibility measure of infeasible target values is now displayed.
- Graphs and tables have been added to the pages dedicated to each problem:
  they show a focus on the performance measure near the target values,
  the execution time,
  the infeasibility measure,
  and the number of unsatisfied constraints.
- Pages dedicated to the results of each algorithm configuration on each problem
  have been added.
  They feature graphs and tables representing the performance measure,
  the infeasibility measure, and the number of unsatisfied constraints.
- The scale of the axis showing the number of function evaluations
  can now be made logarithmic.

## Changed

- The phrasing "problem configuration" is now used instead of "benchmarking problem".
  Thus the two main types of inputs to be defined by the user
  are *algorithm* configurations and *problem* configurations.

#### Problems

- The class to implement optimization benchmarking problems is now called
  ``OptimizationBenchmarkingProblem`` (rather than ``Problem`` formerly).

#### Benchmarker

- Argument ``databases_path`` of ``Benchmarker.__init__`` is renamed into ``hdf_path``
  as the saved files could represent caches rather than databases.
- Arguments ``problems`` and ``algorithm`` of ``Benchmarker.execute`` are renamed into
  ``problem_configurations`` and ``algorithm_configurations``
  to avoid confusion with optimization problems and algorithm names respectively.
- Argument ``number_of_processes`` of ``Benchmarker.execute`` is renamed into
  ``n_processes`` for consistency with GEMSEO.
- The stopping criteria of the algorithms are no longer automatically disabled.
  The user is now free to disable (or not) the stopping criteria of their choice
  in the options of the algorithm configurations.

#### Report

- The results on each problem are now displayed on separate pages
  rather than on the page of the problems group.
- Setting the optimum of a problem is no longer mandatory.
- The performance histories returned by ``PerformanceHistory.compute_cumulated_minimum``
  and ``PerformanceHistory.extend`` now contain copies of history items
  rather than replications of the same objects.

### Scenario

- Argument ``number_of_processes`` of ``Scenario.execute`` is renamed into
  ``n_processes`` for consistency with GEMSEO.

## Fixed

#### Results

- Removing leading infeasible items from an infeasible performance history
  now returns an empty performance history.
- Path options are now properly supported.

#### Report

- The performance measures and target values of maximization problems
  are now correctly displayed instead of being treated as minimization data.
- Negative performance measures are now properly represented on logarithmic scales.

#### Benchmarker

- When overwriting histories,
  the paths already in the ``Results`` are now effectively removed.
- When threading, a log file is written in the performance history directory.
- When multiprocessing, a log file is written next to each performance history.

#### Scenario

- An algorithm configuration can now belong to several groups
  of algorithm configurations.

### Removed

#### Problems

- Method ``Problem.plot_histories`` was removed as it was redundant with ``Figures.plot``.
  To obtain a figure similar to the one formerly returned by

  ```python
  problem.plot_histories(
    algos_configurations,
    results,
    False,
    file_path,
    plot_all_histories,
    alpha,
    markevery,
    infeasibility_tolerance,
    max_eval_number,
    use_log_scale
  )
  ```

  one can make the following call instead:

  ```python
  Figures(
    algos_configurations,
    ProblemsGroup(problem.name, [problem]),
    results,
    file_path.parent,
    infeasibility_tolerance,
    max_eval_number,
    {"alpha": alpha, "markevery": markevery}
  ).plot(plot_all_histories, use_log_scale, False, False, False)
  ```

- Method ``Problem.compute_performance`` was removed as is was redundant with ``PerformanceHistory.from_problem``.
  To obtain values similar to the former

  ```python
  objective_values, infeasibility_measures, feasibility_statuses = Problem.compute_performance(problem)
  ```

  one can use the following instructions instead:

  ```python
  performance_history = PerformanceHistory.from_problem(problem)
  objective_values = performance_history.objective_values
  infeasibility_measures = performance_history.infeasibility_measures
  feasibility_statuses = [item.is_feasible for item in performance_history]
  ```

### Results

- Methods ``PerformanceHistories.plot_algorithm_histories``
  and ``PerformanceHistory.plot`` were removed
  as they were redundant with ``Figures.plot``.
  To obtain a figure similar to the one formerly returned by

  ```python
  performance_histories.plot_algorithm_histories(
    axes,
    algorithm_name,
    max_feasible_objective,
    plot_all,
    color,
    marker,
    alpha,
    markevery
  )
  ```

  one can use the following instructions instead:

  ```python
  results = Results()
  for index, performance_history in enumerate(performance_histories):
    path = f"{index}.json"
    performance_history.to_file(path)
    results.add_path(algorithm_name, problem.name, path)

  Figures(
    AlgorithmsConfigurations(AlgorithmConfiguration(algorithm_name)),
    ProblemsGroup(problem.name, [problem]),
    results,
    ".",
    0,
    0,
    {"color": color, "marker": marker, "alpha": alpha, "markevery": markevery}
  ).plot(plot_all, False, False, False, False)
  ```

## Version 3.0.0 (November 2024)

### Added

#### Benchmarker

- The option ``log_gemseo_to_file`` has been added to ``Benchmarker.execute``
  and ``Scenario.execute`` to save the GEMSEO log of each algorithm execution
  to a file in the same directory as its performance history file.

#### Data profiles

- Target values can be plotted on existing axes as horizontal lines with
  ``TargetValues.plot_on_axes``.

#### Results

- The distribution of a collection of performance histories can be plotted in terms of
  performance measure (```PerformanceHistories.plot_performance_measure_distribution``),
  infeasibility measure (```PerformanceHistories.plot_infeasibility_measure_distribution``)
  and number of unsatisfied constraints
  (```PerformanceHistories.plot_number_of_unsatisfied_constraints_distribution``).

### Changed

#### Results

- Methods
  ``PerformanceHistory.compute_cumulated_minimum``,
  ``PerformanceHistory.extend``,
  ``PerformanceHistory.remove_leading_infeasible``,
  and ``PerformanceHistory.shorten``
  preserve the attributes other than ``PerformanceHistory.items``.

## Version 2.0.0 (December 2023)

### Changed

#### Benchmarker

- The option to automatically save the logs of pSeven has been removed
  from classes ``Scenario`` and ``Benchmarker``.
  However, the user can still save these logs
  by passing an instance-specific option to ``AlgorithmConfiguration``
  (refer to the "Added" section of the present changelog).
  For example:
  ``instance_algorithm_options
  ={"log_path": lambda problem, index: f"my/log/files/{problem.name}.{index}.log"}``.
  N.B. the user is now responsible for the creation of the parent directories.
- Class ``Worker`` no longer sets ``PerformanceHistory.doe_size``
  to the length of the value of the pSeven option ``"sample_x"``.
  Note that this does not affect the behavior of ``gemseo-benchmark``:
  ``PerformanceHistory.doe_size`` is only used as convenience
  when loading/saving a ``PerformanceHistory`` using a file.
  In particular, the behavior of ``Report`` is not changed.
  The user can still set the value of ``PerformanceHistory.doe_size``
  by themselves since it is a public attribute.

### Added

- Support for Python 3.11.

#### Algorithms

- Algorithm options specific to problem instances (e.g. paths for output files)
  can be passed to ``AlgorithmConfiguration`` in the new argument ``instance_algorithm_options``.

#### Benchmarker

- One can get the path to a performance history file with ``Benchmarker.get_history_path``.

### Removed

- Support for Python 3.8.

## Version 1.1.0 (September 2023)

### Added

#### Results

- The names of functions and the number of variables are stored in the
    performance history files.

#### Report

- The optimization histories can be displayed on a logarithmic scale.

#### Scenario

- The options `custom_algos_descriptions` and
    `max_eval_number_per_group` of `Report`{.interpreted-text
    role="class"} can be passed through `Scenario`{.interpreted-text
    role="class"}.

### Fixed

#### Report

- The sections of the PDF report are correctly numbered.
- The graphs of the PDF report are anchored to their expected
    locations.

## Version 1.0.0 (June 2023)

First version.
