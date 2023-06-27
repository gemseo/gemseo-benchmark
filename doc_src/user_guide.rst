..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

User guide
==========

The :mod:`gemseo_benchmark` package provides functionalities to benchmark optimization
algorithms,
that is, to measure and compare their performances.

A typical use of this package consists of the following steps:

#. define the :ref:`algorithms configurations <algorithms-configurations>` to be compared,
#. define the :ref:`benchmarking problems <benchmarking-problems>` that will serve as landmarks for the analysis,
#. execute a benchmarking :ref:`scenario <scenario>` to produce

   #. the :ref:`results <results>` of the algorithms configurations on the benchmarking problems,
   #. a benchmarking :ref:`report <report>` in HTML or PDF format illustrated with :ref:`data profiles <data-profiles>`.

.. note:: Other algorithms will be supported in the future
   (ex: root-finding algorithms).

The following sections present the sub-packages of :mod:`gemseo_benchmark`.

.. _algorithms-configurations:

Algorithms configurations
-------------------------

The :mod:`~gemseo_benchmark.algorithms` sub-package is responsible for
the definition of the algorithms configurations to be investigated
in a benchmarking study.

An :class:`.AlgorithmConfiguration` contains:

* the name of an algorithm,
* optionally, a name for the configuration
  (it will be generated automatically if unspecified),
* the values passed as its options
  (default values are used for the unspecified options).

For example,
we may consider the L-BFGS-B algorithm with its ``maxcor`` option
(the maximum number of corrections of the Hessian approximation)
set to 2.

.. code-block::

   lbfgsb_2_corrections = AlgorithmConfiguration(
      "L-BFGS-B",
      "L-BFGS-B with 2 Hessian corrections",
      maxcor=2,
   )

Additionally, we may consider the same algorithm with a different option value, say 20.

.. code-block::

   lbfgsb_20_corrections = AlgorithmConfiguration(
      "L-BFGS-B",
      "L-BFGS-B with 20 Hessian corrections",
      maxcor=20,
   )

Of course it is also possible to consider an algorithm with all its options
set to their default values.

.. code-block::

   slsqp_default = AlgorithmConfiguration("SLSQP")

The class :class:`.AlgorithmsConfigurations` is useful to
gather algorithms configurations in groups
so that they be treated together in a :ref:`benchmarking report <report>`.

.. code-block::

    lbfgsb_configurations = AlgorithmsConfigurations(
        lbfgsb_2_corrections,
        lbfgsb_20_corrections,
        name="L-BFGS-B configurations",
    )


.. _benchmarking-problems:

Benchmarking problems
---------------------

The :mod:`~gemseo_benchmark.problems` sub-package handles the benchmarking problems,
on which the performances of the algorithms configurations is to be measured.

A :class:`.Problem` contains the mathematical definition of the problem,
as an :class:`~gemseo.algos.opt_problem.OptimizationProblem`,
and requires three other features.

#. The starting points, from which the algorithms configurations should be launched
   on the problem.
   Indeed, an algorithm may be quite dependent on the starting point.
   Therefore,
   in the context of a benchmarking study,
   it is advised to consider several starting points.

   #. One can pass the starting points directly,
   #. or configure their generation as a design of experiments (DOE).

#. The best objective value known for the problem.

#. The :ref:`target values <target-values>`,
   necessary to compute :ref:`data profiles <data-profiles>`:
   typically, a scale of objective functions values ranging
   from a relatively easily achievable value to the best value known.
   Similarly to starting points,
   the target values can be either passed directly,
   or their generation can be configured.

For example, we define below benchmarking problems based on
:class:`~gemseo.problems.analytical.rastrigin.Rastrigin`
and :class:`~gemseo.problems.analytical.rosenbrock.Rosenbrock` respectively,
where

* 5 starting points are computed by latin hypercube sampling (LHS),

  .. code-block::

     doe_settings = {"doe_size": 5, "doe_algo_name": "lhs"}

* and the target values are passed directly
  as an exponential scale towards the minimum (zero).

  .. code-block::

    target_values = TargetValues([10**-4, 10**-5, 10**-6, 0.0])

(The class :class:`.TargetValues` will be presented
:ref:`further down <target-values>`.)

.. code-block::

   rastrigin = Problem(
       "Rastrigin",
       Rastrigin,
       target_values=target_values,
       **doe_settings,
       optimum=0.0,
   )

   rosenbrock = Problem(
       "Rosenbrock",
       Rosenbrock,
       target_values=target_values,
       **doe_settings,
       optimum=0.0,
   )

Note that the second argument of :class:`.Problem` must be callable.
For example, a five-variables benchmarking problem
based on :class:`~gemseo.problems.analytical.rosenbrock.Rosenbrock`
may be defined as follows.

.. code-block::

  rosenbrock_5d = Problem(
      "Rosenbrock 5D",
      lambda: Rosenbrock(5),
      target_values=target_values,
      **doe_settings,
      optimum=0.0,
  )

The class :class:`.ProblemsGroup` is useful to
gather reference problems in groups
so that they be treated together in a :ref:`benchmarking report <report>`.

.. code-block::

   problems_2D = ProblemsGroup(
       "2-variabbles functions",
       [rastrigin, rosenbrock],
       description="Unconstrained functions depending on 2 variables.",
   )


.. _results:

Results
-------

The :mod:`~gemseo_benchmark.results` sub-package manages the results produced by
the algorithms configurations on the benchmarking problems.

The history of the data produced by an algorithm configuration on a benchmarking problem
is stored in a :class:`.PerformanceHistory`.
More precisely:

* A value of interest in the benchmarking of algorithms is defined and named *performance value*.
  The most telling performance value is
  the value of the objective function for an optimization problem,
  or the value of a residual for a nonlinear equation.
* Each performance value is stored in a :class:`.HistoryItem`,
  along with an infeasibility measure (especially for problems subject to constraints).
* A :class:`.PerformanceHistory` is a sequence of :class:`.HistoryItem`\ s.
  The index of the sequence is understood as the 0-based number of functions
  evaluations.

A :class:`.PerformanceHistory` may be saved to a file
in `JavaScript Object Notation (JSON) <https://www.json.org>`_.

The class :class:`.Results` gathers the paths to each :class:`.PerformanceHistory`
in a benchmarking study.
In practice,
:class:`.Results` are generated by a :ref:`benchmarking scenario <scenario>`,
thanks to :meth:`.Benchmarker.execute`.


Benchmarker
-----------

The :mod:`~gemseo_benchmark.benchmarker` sub-package is responsible
for the generation of the results.

The class :class:`.Benchmarker` is responsible for two tasks:

#. executing
   (possibly in parallel)
   the algorithms configurations on the reference problems,
#. saving the performance histories to files,
   and storing their paths in :class:`.Results`.


.. _data-profiles:

Datas profiles
--------------

The :mod:`~gemseo_benchmark.data_profiles` sub-package handles
the computation of data profiles.

A *data profile* is a graph that represents the extent to which an algorithm solves a
problem (or a group of problems) for a given number of functions evaluations.
To :ref:`clarify this definition <data-profile>` we need to introduce target values.

.. _target-values:

Target values
^^^^^^^^^^^^^

The difficulty of a benchmarking problem is represented by a scale of performance
values, called *target values*, ranging from a relatively easily achievable value to
the best value known.
The most telling example of target value is the optimal value of the objective function.
Target values can be thought as milestones on the trajectory towards the best value
known.

.. code-block::

   target_values = TargetValues([10**-4, 10**-5, 10**-6, 0.0])

Since a sequence of target values is in fact a sequence of :class:`.HistoryItem`\ s,
the class :class:`.TargetValues` is a subclass of :class:`.PerformanceHistory`.

Targets generator
^^^^^^^^^^^^^^^^^

The target values of a problem can be handpicked but they can also be automatically
computed with a generator of target values.

A :class:`.TargetsGenerator` relies on algorithms chosen as references.

#. The problem is solved with the reference algorithms from each starting point.
#. Instances of :class:`.PerformanceHistory` representing the history of the best
   performance value (which is decreasing) are computed,
   e.g. :math:`\{\min_{0\leq i \leq k} f(x_i)\}_{0 \leq k \leq K}`
   where :math:`f` is the objective function
   and :math:`x_k` are the values of the design variables at iteration :math:`k`.
#. A notion of *median history* is computed from these histories.
#. Performance values are picked at uniform intervals in the median history:
   these are the target values.

.. _data-profile:

Data profile
^^^^^^^^^^^^

The *data profile* of an algorithm relative to a benchmarking problem
(or a group of benchmarking problems)
is the graph representing the ratio of target values reached by the algorithm
relative to the number functions evaluations performed by the algorithm.

.. figure:: _static/data_profiles.png
   :alt: Three data profiles.

   The data profiles of three algorithms.


.. _report:

Report
------

The :mod:`~gemseo_benchmark.report` sub-package manages the automatic generation of a
benchmarking report in PDF or HTML format describing:

* the algorithms configurations,
* the benchmarking problems,
* the results generated by the algorithms on the problems,
  especially in the form of data profiles.


.. _scenario:

Scenario
--------

The class :class:`.Scenario` is the highest-level class of the package:
it lets the user execute the algorithms configurations on the problems
and generate a benchmarking report by calling a single method.

.. code-block::

   scenario_dir = Path("scenario")
   scenario_dir.mkdir()
   scenario = Scenario([lbfgsb_configurations], scenario_dir)
   results = scenario.execute([problems_2D])
