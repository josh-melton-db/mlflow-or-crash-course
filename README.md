## MLflow OR Crash Course

This repo is a compact companion for a Medium-style post about treating an operations research workflow like any other MLflow project:

1. Define a small optimization problem.
2. Benchmark multiple solver libraries and parameter settings.
3. Track the comparison in MLflow.
4. Pick the champion.
5. Package the winner as an MLflow Model From Code.
6. Deploy it to Databricks Model Serving on serverless compute.

The example problem is a capital allocation / project portfolio optimizer. Each candidate project has an expected value, a cost, and an implementation-hours footprint. The model chooses the best subset of projects under budget and capacity constraints.

### What gets compared

- `OR-Tools CP-SAT`
- `SciPy milp` (HiGHS-backed mixed integer programming)

The benchmark logs:

- average objective value
- average solve time
- feasible ratio
- optimal ratio
- budget and capacity utilization

The selection rule is intentionally simple and blog-friendly: maximize feasible ratio, then maximize average objective, then maximize optimal ratio, then minimize mean solve time.

### Why MLflow fits here

The goal is not to pretend an OR solver is a neural network. The point is that solver experiments still have the same operational questions:

- Which configuration is best?
- What changed between runs?
- Which version should we promote?
- How do we serve the winner reliably?

MLflow gives you one place to track runs, keep artifacts, register the champion, and package a custom inference wrapper using [Models From Code](https://mlflow.org/docs/latest/ml/model/models-from-code/).

### Repo layout

- `src/mlflow_or_crash_course/` contains the optimization logic, MLflow workflow, CLI, and Databricks deployment helpers.
- `scripts/deploy_databricks.py` deploys the bundle, runs the benchmark job, and updates the serving endpoint.
- `resources/` contains the Databricks bundle resources for Unity Catalog bootstrap and the serverless job.
- `tests/` contains focused solver and workflow smoke tests.

### Local quickstart

Create the environment and run tests:

```bash
uv run pytest
```

Run a local benchmark against a local MLflow file store:

```bash
uv run mlflow-or-crash-course benchmark \
  --experiment-name mlflow-or-crash-course-local \
  --tracking-uri file:./mlruns \
  --registry-uri "" \
  --scenario-count 4 \
  --output-dir build/generated
```

Print a sample serving payload:

```bash
uv run mlflow-or-crash-course sample-request
```

### Databricks deployment

This project is already configured for the `azure` Databricks CLI profile and uses:

- a serverless Databricks job for benchmarking and model registration
- Unity Catalog for the schema and registered model
- Databricks Model Serving for the final endpoint

Deploy and run the full workflow:

```bash
uv run python scripts/deploy_databricks.py \
  --profile azure \
  --target azure \
  --catalog main \
  --schema or_blog_josh_melton \
  --registered-model-name main.or_blog_josh_melton.portfolio_optimizer \
  --endpoint-name portfolio-optimizer-endpoint
```

That script does three things:

1. `databricks bundle deploy` to create the schema, registered model, and serverless benchmark job.
2. `databricks bundle run benchmark_or_solvers` to benchmark the solver configs and register the winning model version in MLflow.
3. Create or update a Model Serving endpoint that points at the `Champion` model alias.

It also forces the Databricks bundle `direct` engine so the workflow avoids the Terraform download/signature issue currently present with Databricks CLI `0.292.x`.

### Served model input shape

The deployed endpoint expects one row per scenario, with list-like fields serialized as JSON:

- `scenario_id`
- `project_ids_json`
- `values_json`
- `costs_json`
- `hours_json`
- `budget`
- `capacity`

The output contains the chosen projects, objective value, feasibility flags, and aggregate totals.

### Suggested blog outline

- Start with the idea that optimization experiments deserve the same rigor as ML experiments.
- Use the project portfolio problem to compare `OR-Tools` and `SciPy milp`.
- Show the MLflow run comparison, promote the champion, and finish with the served endpoint.
