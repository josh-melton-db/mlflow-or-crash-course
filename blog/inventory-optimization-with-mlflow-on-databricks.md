---
title: "OR-Ops for Inventory Optimization with MLflow on Databricks"
slug: "inventory-optimization-mlflow-databricks"
summary: "Use Databricks notebooks to benchmark OR solvers for inventory replenishment, track objective and runtime metrics in MLflow, promote the winning solver, run Spark batch optimization, and optionally deploy it with serverless model serving."
---

# OR-Ops for Inventory Optimization with MLflow on Databricks

Operations research projects often get treated like one-off analytics exercises.

Someone writes a solver, tunes a few parameters, shares a screenshot of the best run, and moves on.

That works for a prototype. It breaks down quickly when you want repeatability, traceability, and a clean path from experimentation to deployment.

Machine learning teams solved this problem years ago with experiment tracking, model registration, and serving workflows. My view is that optimization workflows deserve the same discipline.

In this post, I will walk through a notebook-first OR-ops pattern for doing exactly that on Databricks with MLflow:

1. Model a supply-chain inventory replenishment problem.
2. Compare `OR-Tools CP-SAT`, `SciPy milp`, and optionally NVIDIA cuOpt.
3. Track the benchmark in MLflow.
4. Choose a champion configuration using business-friendly metrics.
5. Package the winner as an MLflow Model From Code.
6. Run Spark-native batch optimization from the Champion model.
7. Optionally deploy it to Databricks Model Serving on serverless compute.
8. Query the promoted optimization model from Spark, Python, or SQL.

The companion implementation lives in Databricks source notebooks:

- `notebooks/inventory_optimization_crash_course.py`
- `notebooks/inventory_optimization_cuopt_gpu.py`

The CPU notebook is the primary walkthrough. The cuOpt notebook mirrors the same lifecycle and highlights the small set of GPU-specific implementation differences.

## Why inventory replenishment?

Inventory optimization is a great teaching example because it is both familiar and realistic.

The business question is simple:

> Given current inventory, forecast demand, procurement budget, and storage capacity, how many cases of each SKU should we order this week?

That gives us a problem that is easy to explain in prose but still rich enough to benchmark multiple solver strategies.

In the notebook, each SKU has:

- on-hand inventory
- forecast demand
- unit cost
- unit margin
- holding cost
- stockout penalty
- storage units per case
- maximum reorder quantity

The optimizer chooses integer reorder quantities that maximize value while respecting business constraints.

## The optimization model

At a high level, the decision variables are:

- `order_cases_i`: how many cases to reorder for SKU `i`
- `sell_cases_i`: how many cases of demand we can fulfill
- `ending_inventory_i`: how much inventory remains after sales
- `shortage_cases_i`: how much demand goes unfulfilled

The constraints are:

- procurement spend must stay within budget
- ordered volume must stay within storage capacity
- fulfilled demand plus shortage must equal forecast demand
- on-hand inventory plus ordered cases must equal fulfilled demand plus ending inventory

The objective function is:

```text
maximize
  sum(
    sell_cases_i * unit_margin_i
    - ending_inventory_i * holding_cost_i
    - shortage_cases_i * stockout_penalty_i
  )
```

I like this form because it is easy to explain:

- selling into demand is good
- leftover inventory is mildly bad
- stockouts are very bad

That maps nicely to how supply-chain teams think about service level and economics.

## Why a notebook-first workflow?

A single Databricks notebook has a few advantages for this kind of workflow:

- readers can run it top to bottom without mentally reconstructing a package layout
- the narrative and the code live in the same place
- widgets make deployment parameters obvious
- the same notebook can benchmark, register, and optionally deploy the champion
- the same Champion artifact can power Spark batch, local validation, deployed endpoint calls, and SQL access

The notebook starts with lightweight widgets for:

- `catalog`
- `schema`
- `experiment_name`
- `model_name`
- `endpoint_name`
- `scenario_count`
- `seed`
- `deploy_endpoint`

That keeps the flow configurable without turning the post into a config-management tutorial.

## The OR-ops data contract

The notebook keeps two input shapes because they serve different operational needs.

`inventory_sku_inputs` is the normalized table: one row per SKU per planning scenario. That is the shape you want for ETL, BI, joins to product dimensions, audit, and Spark grouped batch optimization.

`inventory_scenario_requests` is the request snapshot table: one row per planning scenario, with arrays for SKU IDs, inventory, forecast, cost, capacity, and penalty inputs. That is the shape you want for exact replay, Model Serving, and SQL `ai_query`.

Those shapes are equivalent from the solver's point of view. A grouped batch function can assemble the request arrays from SKU rows, and a SQL query can call the endpoint from the request snapshot. The MLflow model stays the same either way.

## Step 1: Generate realistic benchmark scenarios

The notebook generates synthetic inventory scenarios for a distribution center rather than relying on a static CSV.

That keeps the example reproducible while still feeling realistic. For each scenario, it samples:

- SKU count
- forecast demand
- current on-hand inventory
- procurement economics
- storage footprint
- maximum reorder quantities

The benchmark sweep uses a range of scenario sizes so we can compare solver behavior as the problem grows.

In the implementation, the small benchmark uses the same SKU counts in both the CPU and cuOpt notebooks:

```python
small_sku_counts = [18, 36, 54, 72]
```

This matters because a solver comparison is only useful when each runtime solves the same generated scenarios. The notebooks also include optional large-scale stress benchmarks. The first uses one larger replenishment scenario, defaults to 2,500 SKUs, and gives each solver the same time budget. The second uses a sparse two-echelon distribution network with source-to-DC flows, DC-to-store flows, product/store demand, capacity constraints, and shortage penalties. Both are logged as separate MLflow experiments because their purpose is not Champion promotion; they compare CPU and GPU behavior under operational SLAs.

## Step 2: Start with a small, explainable example

Before running the full benchmark, the notebook solves a smaller example scenario and displays:

- the input SKU table
- the recommended order plan
- a compact summary row with objective value, fill rate, and utilization metrics

It gives the reader something concrete before introducing the benchmark table:

- which SKUs got reordered
- which SKUs remained constrained
- how much budget was used
- where shortages remained

That small example is the bridge between “here is the business problem” and “here is the experiment framework.”

## Step 3: Benchmark OR-Tools and SciPy

The notebook compares two solver backends:

- `OR-Tools CP-SAT`
- `SciPy milp` backed by HiGHS

Each backend runs with multiple parameter settings:

```python
solver_configs = [
    {
        "name": "ortools_single_thread",
        "library": "ortools_cp_sat",
        "params": {"time_limit_s": 4.0, "num_workers": 1, "relative_gap": 0.0},
    },
    {
        "name": "ortools_parallel",
        "library": "ortools_cp_sat",
        "params": {"time_limit_s": 4.0, "num_workers": 8, "relative_gap": 0.0},
    },
    {
        "name": "scipy_fast_gap",
        "library": "scipy_milp",
        "params": {"time_limit_s": 4.0, "mip_rel_gap": 0.02, "presolve": True},
    },
    {
        "name": "scipy_exact",
        "library": "scipy_milp",
        "params": {"time_limit_s": 8.0, "mip_rel_gap": 0.0, "presolve": True},
    },
]
```

That is enough variation to make the comparison interesting without overwhelming the reader.

The notebook logs scenario-level results and summary metrics for each configuration.

The metrics I care about here are:

- average objective value
- objective decomposition: gross margin reward, holding cost penalty, and stockout penalty cost
- average, p50, p95, and max solve time
- feasible ratio
- optimal ratio
- average fill rate
- shortage rate
- budget utilization
- storage utilization

This is the key point: the benchmark is not just about “highest objective wins.”

For an operational workflow, a slightly lower objective with consistently fast solve time and high fill rate may be the better production choice.

## Step 4: Track the comparison in MLflow

Once the benchmark runs, MLflow becomes the control plane for the experiment.

The notebook logs:

- top-level run parameters such as seed and scenario count
- nested runs for each solver configuration
- metrics for each configuration, including objective subcomponents and solve-time percentiles
- scenario-level comparison tables
- an input example and example solution artifact
- a JSON artifact describing the champion

For the optional large benchmark, the notebooks write comparable CPU and GPU outputs to Delta tables:

- `inventory_large_benchmark_sku_inputs`
- `inventory_large_benchmark_results`
- `inventory_large_benchmark_recommendations`

Those tables make it easy to inspect the best feasible plan, solve time, and recommendation rows side by side without mixing stress-test results into the small Champion-selection run.

For the optional network benchmark, the notebooks write a separate set of CPU and GPU outputs:

- `inventory_network_benchmark_inputs`
- `inventory_network_benchmark_lanes`
- `inventory_network_benchmark_results`
- `inventory_network_benchmark_flows`

That benchmark is the better place to test whether cuOpt has an advantage, because the formulation is a large sparse LP instead of a compact replenishment MILP.

The notebook keeps the MLflow pattern intentionally small:

```python
summary_frame, scenario_frames = run_benchmark(solver_configs, benchmark_scenarios)

for config in solver_configs:
    summary = summary_frame[summary_frame["config_name"] == config["name"]].iloc[0].to_dict()
    log_solver_run(config, scenario_frames[config["name"]], summary)

champion_row = select_champion(summary_frame)
```

The champion rule maps well to how a supply-chain team would explain a decision:

1. Prefer configurations that actually solve reliably.
2. Prefer higher service level.
3. Prefer stronger business value.
4. Break ties with speed.

That is a much better story than “the solver with the biggest objective won.”

## Step 5: Promote the champion as an MLflow Model From Code

This is where the workflow starts to feel more like modern ML systems than ad hoc optimization scripts.

After selecting a champion, the notebook renders a small Python model file from a checked-in template so the deployment artifact stays readable and version-controlled.

The model keeps the input contract explicit with a small Pydantic request model inside the pyfunc. The public serving signature stays tabular so REST calls and SQL `ai_query` can both use `dataframe_records`.

Then it logs that model to MLflow using Models From Code:

```python
model_info = mlflow.pyfunc.log_model(
    name=model_name,
    python_model=str(model_script_path),
    registered_model_name=registered_model_name,
    input_example=[example_request],
    pip_requirements=build_model_requirements(champion_config["library"]),
)
```

This is one of my favorite parts of the pattern.

Why?

Because the deployed logic stays readable.

Instead of pickling an opaque object graph, Models From Code keeps the actual solver wrapper as inspectable Python. That is especially nice for OR workflows, where the deployment artifact is often custom business logic rather than a standard ML estimator.

After logging the model, the notebook:

- finds the newly registered model version
- assigns the `Champion` alias
- validates the model locally with the same example request
- runs batch optimization from the registered model
- leaves endpoint deployment to a separate cell

The cuOpt companion uses the same model template, but packages one additional helper file:

- `notebooks/model_code/cuopt_inventory_subprocess.py`

That helper keeps native GPU solver execution isolated while still making the deployed code inspectable.

## Step 6: Run Spark batch optimization

The most common production path for an OR model is not an endpoint. It is a scheduled batch solve:

1. Read the latest planning inputs from Delta.
2. Group by the unit of optimization, such as `scenario_id`.
3. Assemble the scenario request.
4. Load the MLflow `Champion` model.
5. Solve each scenario in parallel across Spark tasks.
6. Write scenario-level outcomes and SKU-level recommendations back to Delta.

The CPU notebook implements that pattern with `applyInPandas` over `inventory_sku_inputs`. The grouped function loads `models:/<catalog>.<schema>.<model>@Champion` once per Python worker, calls `predict([request])`, and returns one recommendation row per SKU.

The outputs are:

- `inventory_optimization_scenario_results`: one row per solved scenario
- `inventory_optimization_recommendations`: one row per SKU recommendation with repeated scenario-level metrics

This is the tactical OR-ops distinction I would emphasize: Model Serving is excellent for applications and what-if calls, but Spark batch is usually the better way to optimize thousands of planning scenarios that already live in Delta.

For cuOpt, the same idea applies but the runtime constraints are different. If you need high-throughput GPU batch solving, use GPU-compatible workers or persistent GPU actors so the CUDA context stays warm. Otherwise, route request snapshots through the GPU Model Serving endpoint.

## Step 7: Deploy to Databricks Model Serving

If `deploy_endpoint` is set to `true`, the notebook creates or updates a serverless model serving endpoint in a separate step from the experiment run.

That split is useful because benchmarking and registration are fast enough to rerun often, while endpoint deployment can take longer and tends to be the operational step you rerun independently.

The deployed endpoint name is:

- `inventory-optimizer-endpoint`

The registered model lives in Unity Catalog, for example:

- `demos.default.inventory_optimization`

In the notebook, the serving endpoint is wired using the Databricks SDK and a single routed served entity. Once the endpoint is ready, it becomes a clean JSON interface for replenishment recommendations.

One practical note: creating or updating a serverless serving endpoint can easily take up to 20 minutes, which is exactly why it helps to keep it in its own cell.

## What the request and response look like

The request body uses `dataframe_records` with one record per scenario. The array fields still pass directly; they are not serialized into JSON strings.

Here is a small example request:

```json
{
  "dataframe_records": [
    {
      "scenario_id": "dc_week_01",
      "sku_ids": ["SKU_A", "SKU_B", "SKU_C"],
      "on_hand": [10, 6, 1],
      "forecast": [25, 12, 18],
      "unit_cost": [8, 14, 11],
      "unit_margin": [6, 9, 7],
      "holding_cost": [1, 1, 1],
      "stockout_penalty": [8, 10, 9],
      "storage_units": [1, 2, 1],
      "max_order": [20, 10, 25],
      "budget": 220,
      "storage_capacity": 35
    }
  ]
}
```

And here is the kind of response the deployed model returns:

```json
{
  "predictions": [
    {
      "config_name": "ortools_single_thread",
      "fill_rate": 0.7454545454545455,
      "is_feasible": true,
      "is_optimal": true,
      "library": "ortools_cp_sat",
      "objective_value": 142,
      "ordered_sku_count": 2,
      "recommendations": [
        {
          "sku_id": "SKU_A",
          "order_cases": 15,
          "sell_cases": 25,
          "ending_inventory_cases": 0,
          "shortage_cases": 0
        },
        {
          "sku_id": "SKU_C",
          "order_cases": 9,
          "sell_cases": 10,
          "ending_inventory_cases": 0,
          "shortage_cases": 8
        }
      ],
      "scenario_id": "dc_week_01",
      "status": "OPTIMAL",
      "total_order_spend": 219,
      "total_storage_used": 24
    }
  ]
}
```

I like this output shape because it works for both humans and applications:

- the top-level metrics are easy to inspect
- the objective subcomponents explain why one answer is better than another
- the nested order recommendations can be passed downstream to another system

## Step 8: Reuse the Champion from Python and SQL

What I like most about the updated notebook is that it does not stop at batch outputs or deployment.

The same saved example scenario is reused to call the serving endpoint from Python with the same record shape used by SQL `ai_query`:

```python
python_endpoint_response = workspace.api_client.do(
    method="POST",
    path=f"/serving-endpoints/{endpoint_name}/invocations",
    body={"dataframe_records": [python_request]},
)
```

That payload detail matters because `ai_query` invokes custom model endpoints with a tabular record payload. Keeping REST and SQL on `dataframe_records` avoids maintaining two serving contracts for the same OR model.

The notebook also shows the SQL path with `ai_query`:

```sql
SELECT
  scenario_id,
  prediction.result.status AS status,
  prediction.result.objective_value AS objective_value,
  prediction.result.fill_rate AS fill_rate,
  prediction.result.recommendations AS recommendations,
  prediction.errorMessage AS error_message
FROM (
  SELECT
    scenario_id,
    ai_query(
      endpoint => 'inventory-optimizer-endpoint',
      request => named_struct(
        'scenario_id', scenario_id,
        'sku_ids', sku_ids,
        'on_hand', on_hand,
        'forecast', forecast,
        'unit_cost', unit_cost,
        'unit_margin', unit_margin,
        'holding_cost', holding_cost,
        'stockout_penalty', stockout_penalty,
        'storage_units', storage_units,
        'max_order', max_order,
        'budget', budget,
        'storage_capacity', storage_capacity
      ),
      returnType => 'STRUCT<scenario_id:STRING,status:STRING,objective_value:DOUBLE,fill_rate:DOUBLE,recommendations:ARRAY<STRUCT<sku_id:STRING,order_cases:BIGINT,sell_cases:BIGINT,ending_inventory_cases:BIGINT,shortage_cases:BIGINT>>>',
      failOnError => false
    ) AS prediction
  FROM demos.default.inventory_scenario_requests
)
```

That is the broader Databricks story: the same governed optimization model can serve Spark batch jobs, application code, and SQL users without copying solver logic into multiple places.

## Why this pattern matters

The most important takeaway is not “OR-Tools beat SciPy” or vice versa.

The important takeaway is that optimization workflows benefit from the same lifecycle tools we already expect in ML:

- tracked experiments
- reproducible comparisons
- registered promoted artifacts
- clean batch and deployment paths
- Spark, Python, and SQL inference from the same promoted artifact

This notebook-first pattern is especially useful when the artifact is not a conventional ML model.

That is exactly where MLflow Models From Code shines. It gives you a way to operationalize custom logic without pretending it is a scikit-learn object.

## Where to go next

This example focuses on weekly replenishment because it is easy to explain, but the same pattern extends naturally to:

- safety stock tuning
- multi-echelon inventory planning
- transportation and routing
- workforce scheduling
- production planning

The core idea stays the same:

> treat optimization experiments like first-class production assets, not like disposable scripts.

If you want the fastest way to try this yourself, start with the notebook in the repo, run the small example scenario, then inspect the MLflow benchmark table before deploying the endpoint.

That is the point where the workflow usually clicks.
