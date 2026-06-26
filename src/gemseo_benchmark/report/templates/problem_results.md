# {{ problem.name }}

The results of the group of algorithms configurations '{{ algorithm_configurations.name }}'
for problem configuration '{{ problem.name }}'.

## Data profiles

![data_profile]({{ figures["data_profile.png"] }})
/// caption
The data profiles of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.
///

## {{ problem.performance_measure_label }}

![performance_measure]({{ figures["performance_measure.png"] }})
/// caption
The {{ problem.performance_measure_label | lower }} of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.
///

![performance_measure_focus]({{ figures["performance_measure_focus.png"] }})
/// caption
The {{ problem.performance_measure_label | lower }} of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.
///

{{ tables["performance_measure.csv"] | csv_to_md_table }}
/// caption
The *final* {{ problem.performance_measure_label | lower }} of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.
///

{% if problem.number_of_scalar_constraints %}

## Infeasibility measure

![infeasibility_measure]({{ figures["infeasibility_measure.png"] }})
/// caption
The infeasibility measure of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.
///

{{ tables["infeasibility_measure.csv"] | csv_to_md_table }}
/// caption
The *final* infeasibility measure of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.
///

## Number of unsatisfied constraints

![number_of_unsatisfied_constraints]({{ figures["number_of_unsatisfied_constraints.png"] }})
/// caption
The number of unsatisfied constraints of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.
///

{{ tables["number_of_unsatisfied_constraints.csv"] | csv_to_md_table }}
/// caption
The *final* number of unsatisfied constraints of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.
///

{% endif %}

## Execution time

![execution_time]({{ figures["execution_time.png"] }})
/// caption
The execution time of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.
///

## Results for each algorithm configuration

{% for algo_config_results in algorithm_configurations_results %}- [{{ algo_config_results | stem }}](<{{ algo_config_results }}>)
{% endfor %}
