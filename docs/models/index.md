{% from "commands.md" import kv_cache, precision with context %}
{%- macro k(n) %}{{ (n // 1024) ~ "K" if n % 1024 == 0 else n }}{% endmacro -%}
{%- macro sweep(w) %}{{ k(w.input) }} in{% if w.output is defined %} / {{ k(w.output) }} out{% endif %} at concurrency {{ w.widths | join(", ") }}{% endmacro -%}
{#- A level's cell: measured on the row itself, on the checkpoint whose layers it shares, or not at all. -#}
{%- macro anchor(m) %}{{ m.doc }}.md#{{ m.id | lower | replace("/", "--") | replace(".", "-") }}{% endmacro -%}
{%- macro cell(model, measured, own) -%}
{%- if not own %}—{% elif measured is sameas model %}✓{% else %}as [{{ measured.id.split("/")[1] }}]({{ anchor(measured) }}){% endif -%}
{%- endmacro -%}
# Models

These checkpoints run on RNGD at the speed CI measures for them.
A model is listed only while it runs in CI, so a listed model runs at its measured speed, not merely runs.

## Available models

{% for dashboard in dashboards.values() %}
{%- set rows = benchmarks.models | selectattr("task", "in", dashboard.tasks) | list %}
{%- set tasks = rows | map(attribute="task") | unique | list %}
### {{ dashboard.title }}

{%- set ns_types = namespace(seen=[]) %}
{%- for model in rows %}{% set t = model_types["models/" ~ model.doc ~ ".md"] %}{% if t not in ns_types.seen %}{% set ns_types.seen = ns_types.seen + [t] %}{% endif %}{% endfor %}

<div class="catalog{{ " aligned" if ns_types.seen | length > 1 }}" markdown>
{%- set ns = namespace(last=none) %}
{%- for model in rows %}
{%- set model_type = model_types["models/" ~ model.doc ~ ".md"] %}
{%- if model_type != ns.last %}
{%- if ns.last is not none %}

{% endif %}
#### {{ model_type }}

| Checkpoint |{% if tasks | length > 1 %} Task |{% endif %} Tier | Weights | KV cache | Kernel and forward | Serving |
|---|{% if tasks | length > 1 %}---|{% endif %}---|---|---|:-:|:-:|
{%- set ns.last = model_type %}
{%- endif %}
{%- set measured = benchmarks.models | selectattr("id", "equalto", model.measured_as | default(model.id)) | first %}
{%- set served = measured.task != model.task or measured is sameas model %}
| [{{ model.id }}]({{ anchor(model) }}) |{% if tasks | length > 1 %} {{ model.task | capitalize }} |{% endif %} <span class="tier tier-{{ model.tier }}">{{ model.tier | capitalize }}</span> | {{ precision(model, "weights") }} | {{ kv_cache(model) }} | {{ cell(model, measured, measured.forward is defined) }} | {{ cell(model, model if served else measured, benchmarks.tasks[model.task].serving is defined) }} |
{%- endfor %}

</div>
{% endfor %}

A cell names the checkpoint a row takes a level from when the two share layers.
The inventory [`.github/benchmarks.json`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/.github/benchmarks.json) feeds these tables, the checkpoints CI downloads and the tests it measures.

{% if nightly_run -%}
## Latest nightly

From the [nightly run]({{ nightly_run.run }}) of {{ nightly_run.date }} at [`{{ nightly_run.commit[:9] }}`](https://github.com/furiosa-ai/furiosa-opt-models/commit/{{ nightly_run.commit }}):
{% for w in benchmarks.tasks.text.serving.nightly %}
{%- set measured = nightly_run.serving | selectattr("scenario", "equalto", w.input ~ "-" ~ w.output) | list %}
{%- if measured %}

| {{ k(w.input) }} in, {{ k(w.output) }} out | Concurrency | TTFT (ms) | TPOT (ms) | Output tokens/s |
| --- | --- | --- | --- | --- |
{% for r in measured | sort(attribute="concurrency") | sort(attribute="model") -%}
| {% for m in benchmarks.models if m.id | lower == r.model %}[{{ m.id }}]({{ m.doc }}.md){% endfor %} | {{ r.concurrency }} | {{ "%.0f" | format(r.ttft_ms) }} | {{ "%.1f" | format(r.tpot_ms) }} | {{ "%.0f" | format(r.output_tokens_per_s) if r.output_tokens_per_s is not none else "—" }} |
{% endfor %}
{%- endif %}
{%- endfor %}
{%- endif %}

## How availability is kept

- <span class="tier tier-gate">Gate</span> checkpoints are measured on every pull request: their forward tests, and a serving smoke test of {{ sweep(benchmarks.tasks.text.serving.gate[0]) }}.
- <span class="tier tier-nightly">Nightly</span> checkpoints, and the gate ones again, are measured once a day on main at every level they have, and the vLLM and SGLang contract tests run on them.
- Each family's dashboard keeps the history of its checkpoints: {% for d in dashboards.values() %}[{{ d.title }}]({{ d.share }}){{ ", " if not loop.last }}{% endfor %}. A model page embeds its checkpoint's.

### Regression gate

A pull request is blocked when a measurement rises above the same measurement of its base commit by more than its level's budget:

| Level | Budget | Measured by |
| --- | --- | --- |
{%- for level, budget in benchmarks.budgets.items() %}
| {{ level | capitalize }} | {{ budget }}% | {{ {"kernel": "Cycle counters of the device, which host scheduling does not reach", "forward": "Host wall clock around one model call"}[level] }} |
{%- endfor %}

A kernel slowdown of more than {{ benchmarks.budgets.kernel }}% blocks even when the forward time stays within its {{ benchmarks.budgets.forward }}% budget.
A measurement that the base commit never recorded has nothing to compare against and does not block.
Serving measurements are recorded but do not block: each cell is one run compared with one run of the base commit, and a saturated cell's time to first token moves by about 15% between runs that leave the engine unchanged.
The pull request's comment lists each measurement over its budget.

## What is measured

Three levels cover three layers of the stack, and a regression is attributed to the lowest level that shows it:

| Level | Measures | Unit |
| --- | --- | --- |
| Kernel | NPU kernel execution, by the profile counter at {{ benchmarks.cycles_per_ms // 1000000 }} GHz | Device cycles, shown in milliseconds |
| Forward | Wall-clock time of one model call, the runtime between kernels included | Milliseconds |
| Serving | Requests through vLLM, by the family's metrics | Milliseconds |

- **Serving.** `vllm bench serve` sweeps each workload; a concurrency runs {{ benchmarks.sweep.warmup_waves }} warm-up wave and {{ benchmarks.sweep.waves }} measured waves of at least {{ benchmarks.sweep.minimum_prompts }} requests. Prompts are random tokens and outputs run to full length, so every run does the same work. CI records medians and skips a workload longer than the checkpoint's context.
    - Text generation: {% for w in benchmarks.tasks.text.serving.nightly %}{{ sweep(w) }}{{ "; " if not loop.last }}{% endfor %}. Time to first token (TTFT) and time per output token (TPOT).
    - Embedding & reranking: {{ sweep(benchmarks.tasks.embedding.serving.nightly[0]) }}, one input or query-document pair a request. Request latency.
- **Phase.** A text checkpoint's forward test times the prompt's last chunk and the decode step halfway through the output, at one request and at each nightly concurrency.
- **Shared layers.** A checkpoint sharing another's layers takes that checkpoint's kernel and forward measurements, and within its task its serving ones too.

[Measurement implementation](../develop/testing.md#measurement-implementation) describes how each level is recorded, and [Reproduce a benchmark](../develop/testing.md#reproduce-a-benchmark) how to run one.

### Reading the dashboards

| Family | Serving | Phase |
| --- | --- | --- |
| Text generation | Throughput against interactivity; TTFT and TPOT | Prefill and decode calls, by total time and device time |
| Embedding & reranking | Throughput against request latency; request latency | — |

- **Trend charts** plot each measurement as its change from the commit measured before it, so one commit's effect reads the same at every scale. A level's budget is a dashed line. Hover a point for both values in milliseconds and the CI run; select it to open the commit. A commit on main shows its pull request's measurement of the same tree, so a pull request merged while behind main leaves no point; the nightly run adds points of main itself.
- **Throughput charts** draw the latest four nightly commits, the newest in color, each a curve over the nightly concurrencies. Throughput is concurrency × 1000 / the median TPOT or request latency, a decode or request rate of the chip; for text it leaves out prompt processing. Interactivity is 1000 / median TPOT, the token rate one request sees. A point that moves down at the same concurrency is slower; a curve that flattens shows throughput stopped growing, not where the bottleneck is.
- **Legends** highlight a series when selected. A chart without data says it has no measurement yet.
