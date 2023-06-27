.. _{{ name }}:

{{ name }}
{{ "=" * name|length }}


.. toctree::
   :maxdepth: 1
   :caption: The results of the group of algorithms configurations "{{ name }}"
             on the groups of problems.

{% for document in documents %}   {{ document }}
{% endfor %}
