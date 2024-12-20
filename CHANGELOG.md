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

## Develop

### Added

#### Data profiles

- The color and marker for each algorithm configuration, as well as the grid options,
  can now be customized thanks to the new arguments ``colors``, ``markers``
  and ``grid_kwargs``
  (see ``DataProfile.plot``, ``Problem.compute_data_profile``,
  and ``ProblemsGroup.compute_data_profile``).

#### Report

- The color and marker for each algorithm configuration can now be customized
  at the initialization of ``Report`` thanks to the new arguments``colors``
  and ``markers``.
- The user can now request that ``Scenario.execute`` or ``Report.generate``
  plot only the median of the performance measure rather than its whole range
  thanks to the new boolean argument ``plot_only_median``.
- On the page dedicated to the benchmarking problems,
  the infeasibility measure of infeasibe target values is now displayed.
- For each problem, a graph of the performance measure focusing on the target values
  has been added below the usual one.

### Changed

#### Report

- The results on each problem are now displayed on separate pages
  rather than on the page of the problems group.

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
