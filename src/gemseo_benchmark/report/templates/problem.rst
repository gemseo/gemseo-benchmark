.. _{{ name }}:

{{ name }}
{{ "=" * name|length }}


Description
-----------

{{ description }}

Optimal feasible objective value: {{ optimum }}.


Target values
-------------
{% for target in target_values %}* {{ target.objective_value }} ({{ 'feasible' if target.is_feasible else 'infeasible with infeasibility measure %e' % target.infeasibility_measure }})
{% endfor %}
