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

{% if problem.constraints_names %}
Infeasibility measure
---------------------

.. figure:: /{{ figures["infeasibility_measure.png"] }}
   :alt: The infeasibility measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.

   The infeasibility measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem ':ref:`{{ problem.name }}`'.


Number of unsatisfied constraints
---------------------------------

.. figure:: /{{ figures["number_of_unsatisfied_constraints.png"] }}
   :alt: The number of unsatisfied constraints of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.

   The number of unsatisfied constraints of algorithm configuration '{{ algorithm_configuration.name }}' for problem ':ref:`{{ problem.name }}`'.

{% endif %}
