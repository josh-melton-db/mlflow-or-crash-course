## MLflow OR Crash Course

This repo is a public companion to a blog post about experimenting with inventory optimization on Databricks with MLflow.

The main artifact is a single Databricks source notebook:

- `notebooks/inventory_optimization_crash_course.py`

That notebook generates supply-chain-flavored replenishment scenarios, benchmarks multiple OR solver settings, logs the comparison to MLflow, registers the winner as an MLflow Model From Code, and can optionally deploy the champion to Databricks Model Serving.

If you want a ready-made narrative outline to pair with the notebook, start with `BLOG_COMPANION.md`.

### The example problem

The business story is a distribution-center replenishment plan.

For each SKU, the optimizer decides how many cases to order this week while balancing:

- expected demand
- procurement budget
- storage capacity
- leftover inventory carrying cost
- stockout penalty

This keeps the math approachable for a blog post while still feeling like an actual inventory optimization workflow instead of a generic knapsack.

### What gets compared

- `OR-Tools CP-SAT`
- `SciPy milp` (HiGHS-backed mixed integer programming)

The notebook tracks:

- average objective value
- average solve time
- feasible ratio
- optimal ratio
- average fill rate
- budget utilization
- storage utilization

The champion rule is intentionally simple and reader-friendly: maximize feasible ratio, then maximize fill rate, then maximize objective value, then minimize solve time.

### Why MLflow fits

Inventory optimization experiments still create the same operational questions as ML experiments:

- Which solver configuration is best?
- Which run produced the best service level and economics?
- Which version should we promote?
- How do we serve the recommendation logic reliably?

MLflow handles the experiment tracking, artifact storage, registration, and custom inference packaging using [Models From Code](https://mlflow.org/docs/latest/ml/model/models-from-code/).

### Repo layout

- `notebooks/inventory_optimization_crash_course.py` is the main tutorial notebook.
- `BLOG_COMPANION.md` gives a blog-ready narrative arc, screenshot list, and talking points.
- `resources/` contains the Databricks bundle resources for the Unity Catalog schema, registered model, and serverless notebook job.
- `scripts/deploy_databricks.py` deploys the bundle and runs the notebook job.
- `databricks.yml` defines the Azure target and notebook job variables.

### Run it on Databricks

The project is configured for the `azure` Databricks CLI profile and uses:

- a serverless notebook job
- Unity Catalog for model registration
- Databricks Model Serving for the optional endpoint deployment

Deploy and run the notebook workflow:

```bash
python scripts/deploy_databricks.py \
  --profile azure \
  --target azure \
  --catalog main \
  --schema inventory_optimization_blog \
  --registered-model-name main.inventory_optimization_blog.inventory_optimizer \
  --endpoint-name inventory-optimizer-endpoint \
  --deploy-endpoint true
```

That helper does two things:

1. `databricks bundle deploy` to push the notebook, schema, registered model, and serverless job.
2. `databricks bundle run inventory_optimization_crash_course` to execute the notebook with the selected parameters.

The notebook itself performs the benchmark, registers the champion model version, and optionally creates or updates the serving endpoint.

The helper also forces the Databricks bundle `direct` engine so the workflow avoids the Terraform download/signature issue currently present with Databricks CLI `0.292.x`.

### Open the notebook directly

If you want the most direct tutorial experience, open `notebooks/inventory_optimization_crash_course.py` in Databricks and run it cell by cell. The notebook has widgets for:

- `catalog`
- `schema`
- `experiment_name`
- `registered_model_name`
- `endpoint_name`
- `scenario_count`
- `seed`
- `deploy_endpoint`

### Suggested blog framing

- Start from the idea that OR experiments deserve the same lifecycle discipline as ML experiments.
- Use weekly inventory replenishment as the motivating supply-chain use case.
- Show how solver settings change fill rate, cost efficiency, and runtime.
- End with the champion solver packaged as an MLflow model and deployed on Databricks serverless infrastructure.
