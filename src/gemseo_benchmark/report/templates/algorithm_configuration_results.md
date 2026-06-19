# {{ algorithm_configuration.name }} on {{ problem.name }}

## Performance measure

![The performance measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.](../../../../{{ figures["performance_measure.png"] }})

![The performance measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.](../../../../{{ figures["performance_measure_focus.png"] }})

{{ tables["performance_measure.csv"] | csv_to_md_table }}
{% if problem.number_of_scalar_constraints %}

## Infeasibility measure

![The infeasibility measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.](../../../../{{ figures["infeasibility_measure.png"] }})

{{ tables["infeasibility_measure.csv"] | csv_to_md_table }}

## Number of unsatisfied constraints

![The number of unsatisfied constraints of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.](../../../../{{ figures["number_of_unsatisfied_constraints.png"] }})

{{ tables["number_of_unsatisfied_constraints.csv"] | csv_to_md_table }}
{% endif %}
