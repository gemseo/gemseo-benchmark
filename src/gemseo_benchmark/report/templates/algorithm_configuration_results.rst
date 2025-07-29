{{ algorithm_configuration.name }} on {{ problem.name }}
{{ "=" * algorithm_configuration.name|length }}===={{ "=" * problem.name|length }}


Performance measure
-------------------

.. figure:: /{{ figures["performance_measure.png"] }}
   :alt: The performance measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.

   The performance measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem ':ref:`{{ problem.name }}`'.

.. figure:: /{{ figures["performance_measure_focus.png"] }}
   :alt: The performance measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.

   The performance measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem ':ref:`{{ problem.name }}`'.

.. csv-table:: The *final* feasible performance measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem ':ref:`{{ problem.name }}`'.
   :file: /{{ tables["performance_measure.csv"] }}
   :header-rows: 1
   :stub-columns: 1
{% if problem.number_of_scalar_constraints %}
Infeasibility measure
---------------------

.. figure:: /{{ figures["infeasibility_measure.png"] }}
   :alt: The infeasibility measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.

   The infeasibility measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem ':ref:`{{ problem.name }}`'.

.. csv-table:: The *final* infeasibility measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem ':ref:`{{ problem.name }}`'.
   :file: /{{ tables["infeasibility_measure.csv"] }}
   :header-rows: 1
   :stub-columns: 1

Number of unsatisfied constraints
---------------------------------

.. figure:: /{{ figures["number_of_unsatisfied_constraints.png"] }}
   :alt: The number of unsatisfied constraints of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.

   The number of unsatisfied constraints of algorithm configuration '{{ algorithm_configuration.name }}' for problem ':ref:`{{ problem.name }}`'.

.. csv-table:: The *final* number of unsatisfied constraints of algorithm configuration '{{ algorithm_configuration.name }}' for problem ':ref:`{{ problem.name }}`'.
   :file: /{{ tables["number_of_unsatisfied_constraints.csv"] }}
   :header-rows: 1
   :stub-columns: 1
{% endif %}
