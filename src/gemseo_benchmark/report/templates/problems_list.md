# Problem configurations

The benchmarking reference problems are listed below.

{% for problem_path in problems_paths %}- [{{ problem_path | stem }}](<{{ problem_path }}>)
{% endfor %}
