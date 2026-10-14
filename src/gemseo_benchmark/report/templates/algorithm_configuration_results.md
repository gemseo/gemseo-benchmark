# {{ algorithm_configuration.name }} on {{ problem.name }}

## Performance measure

![performance_measure]({{ figures["performance_measure.png"] }})
/// caption
The performance measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.
///

![performance_measure_focus]({{ figures["performance_measure_focus.png"] }})
/// caption
The performance measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.
///

{{ tables["performance_measure.csv"] | csv_to_md_table }}
/// caption
The *final* feasible performance measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem {{ problem.name }}.
///

{% if problem.number_of_scalar_constraints %}

## Infeasibility measure

![infeasibility_measure]({{ figures["infeasibility_measure.png"] }})
/// caption
The infeasibility measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.
///

{{ tables["infeasibility_measure.csv"] | csv_to_md_table }}
/// caption
The *final* infeasibility measure of algorithm configuration '{{ algorithm_configuration.name }}' for problem {{ problem.name }}.
///

## Number of unsatisfied constraints

![number_of_unsatisfied_constraints]({{ figures["number_of_unsatisfied_constraints.png"] }})
/// caption
The number of unsatisfied constraints of algorithm configuration '{{ algorithm_configuration.name }}' for problem '{{ problem.name }}'.
///

{{ tables["number_of_unsatisfied_constraints.csv"] | csv_to_md_table }}
/// caption
The *final* number of unsatisfied constraints of algorithm configuration '{{ algorithm_configuration.name }}' for problem {{ problem.name }}.
///

{% endif %}
