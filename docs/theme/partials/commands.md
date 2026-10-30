{%- macro quote(arg) -%}
{%- set safe = "%+,-./0123456789:=@ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" -%}
{%- if arg and not arg | reject("in", safe) | list -%}
{{ arg }}
{%- else -%}
'{{ arg | replace("'", "'\"'\"'") }}'
{%- endif -%}
{%- endmacro -%}
{%- macro vllm_serve(row) -%}
{%- set args = benchmarks.tasks[row.task].serve + row.serve | default([]) -%}
{% if "{checkpoint}" in args | join(" ") %}checkpoint=$(hf download {{ quote(row.id) }} --quiet)
{% endif %}vllm serve {{ quote(row.id) }}
{%- for arg in args %} {% for part in arg.split("{checkpoint}") %}
{%- if not loop.first %}"$checkpoint"{% endif %}{% if part or loop.length == 1 %}{{ quote(part) }}{% endif %}
{%- endfor %}{% endfor %}
{%- endmacro -%}
{%- macro vllm_bench(id, s, width) -%}
{%- set w = benchmarks.sweep -%}
vllm bench serve --model {{ quote(id) }} {{ benchmarks.tasks.text.bench | map("string") | join(" ") }} \
  --random-input-len {{ s.input }} --random-output-len {{ s.output }} --max-concurrency {{ width }} \
  --num-prompts {{ [w.minimum_prompts, w.waves * width] | max }} --num-warmups {{ w.warmup_waves * width }}
{%- endmacro -%}
{%- macro sglang_serve(row) -%}
python -m sglang.launch_server --model-path {{ quote(row.id) }}
{%- for arg in sglang_args(row) %} {{ quote(arg) }}{% endfor %}
{%- endmacro -%}
{%- macro request(row, port) -%}
{%- if row.task == "text" -%}
curl http://127.0.0.1:{{ port }}/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "{{ row.id }}", "prompt": "The capital of Korea is", "max_tokens": 12, "temperature": 0}'
{%- elif row.task == "embedding" -%}
curl http://127.0.0.1:{{ port }}/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "{{ row.id }}", "input": ["The capital of France is Paris."]}'
{%- else -%}
curl http://127.0.0.1:{{ port }}/v1/rerank -H 'Content-Type: application/json' \
  -d '{"query": "What is the capital of France?", "documents": ["Tokyo is the capital of Japan.", "Paris is the capital of France."]}'
{%- endif -%}
{%- endmacro -%}
{%- macro kv_cache(row) -%}
{%- if "serve" not in benchmarks.tasks[row.task] %}—{% else -%}
<span class="pill">BF16</span>{% if "fp8_e4m3" in row.serve | default([]) %} <span class="pill">FP8</span>{% endif %}
{%- endif -%}
{%- endmacro -%}
{#- A row's precision of `part`, `weights` or `activations`, as pills; a row without
    `quantization` runs BF16. -#}
{%- macro precision(row, part) -%}
{%- for value in (row.quantization[part] if row.quantization is defined else ["BF16"]) -%}
<span class="pill">{{ value }}</span>{{ " " if not loop.last }}
{%- endfor -%}
{%- endmacro -%}
