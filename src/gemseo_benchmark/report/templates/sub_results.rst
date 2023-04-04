{{ "#" * problems_group_name|length }}
{{ problems_group_name }}
{{ "#" * problems_group_name|length }}

The results of the group of algorithms configurations ":ref:`{{ algorithms_group_name }}`"
on the group of problems "{{ problems_group_name }}".

The algorithms configurations
*****************************

{% for name in algorithms_configurations_names %}* {{ name }}
{% endfor %}


The problems
************

{{ problems_group_description }}

{% for problem_name in problems_figures %}* :ref:`{{ problem_name }}`
{% endfor %}


Benchmarking results
********************

Global results
==============

The performances of the algorithms on the reference problems of the group
"{{ problems_group_name }}" are represented in the following data profile.

.. figure:: /{{ data_profile }}
   :alt: The data profiles for group "{{ problems_group_name }}".

   The data profiles for group "{{ problems_group_name }}".


Results for each problem
========================
{% for problem_name, figures in problems_figures.items() %}
{{ problem_name }}
{{ "-" * problem_name|length }}

.. figure:: /{{ figures["data_profile"] }}
   :alt: The data profiles for problem :ref:`{{ problem_name }}`.

   The data profiles for problem :ref:`{{ problem_name }}`.

.. figure:: /{{ figures["histories"] }}
   :alt: The performance histories for problem :ref:`{{ problem_name }}`.

   The performance histories for problem :ref:`{{ problem_name }}`.

{% endfor %}
