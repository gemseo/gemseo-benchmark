{{ name }}
{{ "=" * name|length }}


-----------
Description
-----------
{{ description }}
{% for a_problem_name, a_problem_description in problems.items() %}

^^^^^^^^
Problems
^^^^^^^^
{{ a_problem_name }}
   {{ a_problem_description }}
{% endfor %}


------------
Data profile
------------
The performances of the algorithms on the reference problems of the group
"{{ name }}" are represented in the following data profile.

{{ data_profile }}