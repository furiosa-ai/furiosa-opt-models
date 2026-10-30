# A Datadog dashboard for each model family, from the layouts and chart specifications next to
# this file, keyed as the layouts are:
#   jq -s -f .github/dashboard/build.jq .github/benchmarks.json \
#     .github/dashboard/{dashboards,trend,capacity}.json
. as [$inventory, $dashboards, $trend, $capacity]

# A workload's sweep directory and `scenario` tag: its input length, then its output length if any.
| def scenario: [.input, .output | values] | map(tostring) | join("-");

# A workload as its legend reads, such as `1K in / 8K out`.
def name: def k: if . % 1024 == 0 then "\(. / 1024)K" else tostring end;
    "\(.input | k) in" + if .output then " / \(.output | k) out" else "" end;

# Widgets one under another, each a [definition, height] pair.
def stack: [foreach .[] as [$definition, $height] (0; . + $height;
    {definition: $definition, layout: {x: 0, y: (. - $height), width: 12, height: $height}})];

# A line across a trend chart at `$y`.
def level($y; $mark): {data: {values: [{}]}, mark: ({type: "rule"} + $mark), encoding: {y: {datum: $y}}};

# The legend of a chart's series; Vega lays out an untitled legend at infinite size when the
# panel has no data.
def legend($title): .layer |= map(if .name == "series" then .encoding.color.title = $title else . end);

# A capacity chart's axes: a text workload trades tokens per user for tokens per chip, a pooling
# one its request latency for requests per chip.
def axes($metric):
    if $metric == "serving.tpot_ms" then . else
      .layer |= map(if .name == "series" then
        .encoding.x = {field: "value", type: "quantitative", title: "Request latency (ms)", scale: {zero: false}}
        | .encoding.y.title = "Requests/s per chip"
        | .encoding.tooltip = [{field: "commit", title: "Commit"}, {field: "batch", title: "Concurrency"},
            {field: "value", title: "Request latency (ms)", format: ".2f"},
            {field: "throughput", title: "Requests/s per chip", format: ".1f"}]
      else . end)
    end;

# The dashboard of a family, whose tasks' nightly workload swept across the most concurrencies is
# the one its capacity chart draws.
def dashboard($tasks):
    ([$inventory.tasks[$tasks[]].serving.nightly // [] | .[]] | max_by(.widths | length)) as $nightly

    # The panel's specification: its legend, and a level's budget as a dashed line.
    | def specification:
        $inventory.budgets[.metric | split(".")[0]] as $budget
        | if .chart == "capacity" then
            .metric as $metric | $capacity | legend("Nightly · \($nightly | name) · latest commits") | axes($metric)
          else
            (.legend // "Workload") as $legend | $trend | legend($legend)
            | if $budget then .layer += [
                level($budget; {color: "#c41200", strokeDash: [4, 4]}),
                level($budget; {type: "text", text: "budget +\($budget)%", color: "#c41200",
                  align: "right", dy: -6, x: {expr: "width"}})
              ] else . end
          end;

    # A panel of a metric in milliseconds, a series per commit, CI run and tag; a kernel's cycles
    # convert at the inventory's `cycles_per_ms`, its profile counter's rate.
    def panel:
        ([if .chart == "capacity" then "scenario:\($nightly | scenario)" else empty end]
         | ["!stage:sample", "!event:pull_request"] + . + ["$model"] | join(",")) as $scope
        | (["commit", "run", "attempt"] + (if .chart == "capacity" then ["batch"] else .tags end)
           | join(",")) as $group
        | {type: "wildcard", title,
           requests: [{response_format: "timeseries",
             queries: [{data_source: "metrics", name: "query1",
               query: "\(.aggregate // "avg"):furiosa_opt.\(.metric){\($scope)} by {\($group)}.fill(null)"}],
             formulas: [{formula: "query1 / \(if .unit == "cycles" then $inventory.cycles_per_ms else 1 end)",
               alias: "value"}]}],
           specification: {type: "vega-lite", contents: specification}};

    # Six rows a panel, and a row for each group's title.
    .widgets |= ([.[] | [{type: "group", title, layout_type: "ordered",
        widgets: ([.panels[] | [panel, 6]] | stack)}, 6 * (.panels | length) + 1]] | stack);

$dashboards | map_values(
    .tasks as $tasks
    | [$inventory.models[] | select(.task | IN($tasks[])) | .id | ascii_downcase] as $models
    | {id, share, models: $models,
       body: ({title: "furiosa-opt-models · \(.title)",
               description: "Δ% vs previous measured commit · hover: ms, CI run · click point: commit · click legend: highlight",
               layout_type: "ordered", reflow_type: "fixed",
               default_timeframe: {type: "live", unit: "day", value: 7},
               template_variables: [{name: "model", prefix: "model", default: $models[0]}],
               widgets} | dashboard($tasks))})
