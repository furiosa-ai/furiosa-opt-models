{#- The levels CI measures a row at itself: its forward test, and serving unless the row is measured
    as another of its task, which serves the same numbers. -#}
{%- macro levels(row) -%}
{%- set measured = benchmarks.models | selectattr("id", "equalto", row.measured_as | default(row.id)) | first -%}
{%- set own = measured.task != row.task or measured is sameas row -%}
{%- set served = own and benchmarks.tasks[row.task].serving is defined -%}
{{ ((["Kernel", "Forward"] if row.forward | default(false) else [])
  + (["Serving"] if served else [])
  + (["Serving capacity"] if served else [])) | join(" · ") }}
{%- endmacro -%}
{#- The dashboard of a row's family. -#}
{%- macro family(row) -%}
{%- for key, dashboard in dashboards.items() if row.task in dashboard.tasks %}{{ key }}{% endfor -%}
{%- endmacro -%}
