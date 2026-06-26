# {{ problem.name }}

The results of the group of algorithms configurations '{{ algorithm_configurations.name }}'
for problem configuration '{{ problem.name }}'.

## Data profiles

![The data profiles of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.]({{ figures["data_profile.png"] }})

## {{ problem.performance_measure_label }}

![The {{ problem.performance_measure_label | lower }} of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.]({{ figures["performance_measure.png"] }})

![The {{ problem.performance_measure_label | lower }} of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.]({{ figures["performance_measure_focus.png"] }})

{{ tables["performance_measure.csv"] | csv_to_md_table }}
{% if problem.number_of_scalar_constraints %}

## Infeasibility measure

![The infeasibility measure of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.]({{ figures["infeasibility_measure.png"] }})

{{ tables["infeasibility_measure.csv"] | csv_to_md_table }}

## Number of unsatisfied constraints

![The number of unsatisfied constraints of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.]({{ figures["number_of_unsatisfied_constraints.png"] }})

{{ tables["number_of_unsatisfied_constraints.csv"] | csv_to_md_table }}
{% endif %}

## Execution time

![The execution time of group {{ algorithm_configurations.name }} for problem configuration {{ problem.name }}.]({{ figures["execution_time.png"] }})

## Results for each algorithm configuration

{% for algo_config_results in algorithm_configurations_results %}- [{{ algo_config_results | stem }}](<{{ algo_config_results }}>)
{% endfor %}
