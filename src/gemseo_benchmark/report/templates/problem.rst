.. _{{ name }}:

{{ name }}
{{ "=" * name|length }}


Description
-----------

{{ description }}

Optimal objective value: {{ optimum }}.


Target values
-------------
{% for target in target_values %}* {{ target }} (feasible)
{% endfor %}
