# {{ problems_group_name }}

The results of the group of algorithms configurations "{{ algorithms_group_name }}"
on the group of problems "{{ problems_group_name }}".

## The algorithms configurations

{% for name in algorithms_configurations_names %}- {{ name }}
{% endfor %}

## The problems

{{ problems_group_description }}

{% for problem_name in problems_names %}- {{ problem_name }}
{% endfor %}

## Benchmarking results

### Global results

The performances of the algorithms on the reference problems of the group
"{{ problems_group_name }}" are represented in the following data profile.

![The data profiles for group "{{ problems_group_name }}".](../../{{ data_profile }})

### Results for each problem

The results of the algorithms configurations for each problem are linked below.

{% for path in group_problems_paths %}- [{{ path | stem }}](<{{ path }}>)
{% endfor %}
