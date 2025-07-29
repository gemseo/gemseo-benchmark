{{ problem.name }}
{{ "=" * problem.name|length }}

The results of the group of algorithms configurations ':ref:`{{ algorithm_configurations.name }}`'
for problem configuration ':ref:`{{ problem.name }}`'.


Data profiles
-------------

.. figure:: /{{ figures["data_profile.png"] }}
   :alt: The data profiles of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.

   The data profiles of group ':ref:`{{ algorithm_configurations.name }}`' for problem configuration ':ref:`{{ problem.name }}`'.


{{ problem.performance_measure_label }}
{{ "-" * problem.performance_measure_label|length }}

.. figure:: /{{ figures["performance_measure.png"] }}
   :alt: The {{ problem.performance_measure_label.lower() }} of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.

   The {{ problem.performance_measure_label.lower() }} of group ':ref:`{{ algorithm_configurations.name }}`' for problem configuration ':ref:`{{ problem.name }}`'.

.. figure:: /{{ figures["performance_measure_focus.png"] }}
   :alt: The {{ problem.performance_measure_label.lower() }} of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.

   The {{ problem.performance_measure_label.lower() }} of group ':ref:`{{ algorithm_configurations.name }}`' for problem configuration ':ref:`{{ problem.name }}`'.

.. csv-table:: The *final* {{ problem.performance_measure_label.lower() }} of group ':ref:`{{ algorithm_configurations.name }}`' for problem configuration ':ref:`{{ problem.name }}`'.
   :file: /{{ tables["performance_measure.csv"] }}
   :header-rows: 1
   :stub-columns: 1

{% if problem.number_of_scalar_constraints %}
Infeasibility measure
---------------------

.. figure:: /{{ figures["infeasibility_measure.png"] }}
   :alt: The infeasibility measure of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.

   The infeasibility measure of group ':ref:`{{ algorithm_configurations.name }}`' for problem configuration ':ref:`{{ problem.name }}`'.

.. csv-table:: The *final* infeasibility measure of group ':ref:`{{ algorithm_configurations.name }}`' for problem configuration ':ref:`{{ problem.name }}`'.
   :file: /{{ tables["infeasibility_measure.csv"] }}
   :header-rows: 1
   :stub-columns: 1

Number of unsatisfied constraints
---------------------------------

.. figure:: /{{ figures["number_of_unsatisfied_constraints.png"] }}
   :alt: The number of unsatisfied constraints of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.

   The number of unsatisfied constraints of group ':ref:`{{ algorithm_configurations.name }}`' for problem configuration ':ref:`{{ problem.name }}`'.

.. csv-table:: The *final* number of unsatisfied constraints of group ':ref:`{{ algorithm_configurations.name }}`' for problem configuration ':ref:`{{ problem.name }}`'.
   :file: /{{ tables["number_of_unsatisfied_constraints.csv"] }}
   :header-rows: 1
   :stub-columns: 1
{% endif %}
Execution time
--------------

.. figure:: /{{ figures["execution_time.png"] }}
   :alt: The execution time of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.

   The execution time of group ':ref:`{{ algorithm_configurations.name }}`' for problem configuration ':ref:`{{ problem.name }}`'.


Results for each algorithm configuration
----------------------------------------

.. toctree::
   :maxdepth: 1

{% for algo_config_results in algorithm_configurations_results %}   {{ algo_config_results }}
{% endfor %}
