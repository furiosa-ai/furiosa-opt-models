{% from "levels.md" import levels, family with context %}
{% from "commands.md" import vllm_serve, sglang_serve, request, kv_cache, precision with context %}
{#- Inventory rows whose `doc` names the including page; a page with none fails the build. -#}
{%- set rows = benchmarks.models | selectattr("doc", "equalto", page.file.name) | list -%}
{%- if not rows %}{{ no_inventory_row_names_this_page }}{% endif %}
{% for row in rows %}
{#- A row measured as another compiles to that row's kernels; one of its task also serves that row's
    numbers, so its page shows that row's charts. -#}
{%- set measured = benchmarks.models | selectattr("id", "equalto", row.measured_as | default(row.id)) | first %}
{%- set shown = measured if measured.task == row.task else row %}
{#- A task whose family has no dashboard shows no charts. -#}
{%- set dashboard = dashboards.get(family(row)) %}
{%- set url = dashboard.share ~ "?tpl_var_model%5B0%5D=" ~ shown.id | lower | urlencode | replace("/", "%2F") if dashboard else none %}

## {{ row.id }} <span class="tier tier-{{ row.tier }}">{{ row.tier | capitalize }}</span> { #{{ row.id | lower | replace("/", "--") | replace(".", "-") }} data-toc-label="{{ row.id }}" }

{{ ("Text generation" if row.task == "text" else row.task | replace("-", " ")) | capitalize }} · Weights {{ precision(row, "weights") }} · Activations {{ precision(row, "activations") }}
{%- if "serve" in benchmarks.tasks[row.task] %} · KV cache {{ kv_cache(row) }}{% endif %}
{%- if nightly_run %} · Verified by the [nightly run]({{ nightly_run.run }}) of {{ nightly_run.date }}{% endif %}
{: .facts }

{% if "serve" in benchmarks.tasks[row.task] -%}
=== "vLLM"

    ```bash
    {{ vllm_serve(row) | indent(4) }}
    ```

    ```bash
    {{ request(row, 8000) | indent(4) }}
    ```
{% endif -%}
{% if sglang_args(row) is not none %}
=== "SGLang"

    ```bash
    {{ sglang_serve(row) | indent(4) }}
    ```

    ```bash
    {{ request(row, 30000) | indent(4) }}
    ```
{% elif "serve" in benchmarks.tasks[row.task] %}
No SGLang command serves this checkpoint yet.
{% endif %}

{%- set own = levels(shown) if dashboard else "" %}
{%- set layers = "Kernel · Forward" if shown is not sameas measured and measured.forward is defined else "" %}
{% macro tags(levels) %}{% for level in levels.split(" · ") %}<span class="level">{{ level }}</span> {% endfor %}{% endmacro %}
{% if own or layers %}
Benchmarks:

{% if own -%}
- {{ tags(own) }}· [Dashboard]({{ url }}){% if shown is not sameas row %} · as [{{ shown.id }}]({{ shown.doc }}.md){% endif %}
{% endif -%}
{% if layers -%}
- {{ tags(layers) }}· as [{{ measured.id }}]({{ measured.doc }}.md)
{% endif %}
{% if own %}
<iframe title="{{ shown.id }} performance history" src="{{ url }}" width="100%" height="{{ 200 + (dashboard.widgets | length) * 80 + (dashboard.widgets | map(attribute="panels") | map("length") | sum) * 440 }}" style="border: 0"></iframe>
{% endif %}
{% else %}
Benchmarks: none; CI runs the framework tests only.
{% endif %}
{% endfor %}
