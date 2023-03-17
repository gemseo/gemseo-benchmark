{{ "#" * name|length }}
{{ name }}
{{ "#" * name|length }}


Description
***********

{{ description }}


Problems
========
{% for problem_name in problems_figures %}* :ref:`{{ problem_name }}`
{% endfor %}


Benchmarking results
********************

Global results
==============

The performances of the algorithms on the reference problems of the group
"{{ name }}" are represented in the following data profile.

.. figure:: /{{ data_profile }}
   :alt: The data profiles for group {{ name }}.

   The data profiles for group {{ name }}.


Results for each problem
========================
{% for problem_name, figures in problems_figures.items() %}
{{ problem_name }}
{{ "-" * problem_name|length }}

.. figure:: /{{ figures["data_profile"] }}
   :alt: The data profiles for problem {{ problem_name }}.

   The data profiles for problem {{ problem_name }}.

.. figure:: /{{ figures["histories"] }}
   :alt: The performance histories for problem {{ problem_name }}.

   The performance histories for problem {{ problem_name }}.

{% endfor %}
