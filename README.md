## MLflow OR-Ops Crash Course

This repo walks through inventory optimization on Databricks as a tactical guide to **OR-ops**: the MLOps-style lifecycle for operations research models.

The main artifacts are Databricks source notebooks:

- `notebooks/inventory_optimization_crash_course.py`
- `notebooks/inventory_optimization_cuopt_gpu.py` is the NVIDIA cuOpt companion notebook for Databricks serverless GPU compute.

The CPU notebook generates supply-chain-flavored replenishment scenarios, benchmarks multiple OR solver settings, logs the comparison to MLflow, registers the winner as an MLflow Model From Code, runs Spark-native batch optimization, and can optionally deploy the champion to Databricks Model Serving. Both notebooks also include optional CPU-vs-GPU stress benchmarks for testing solver behavior under a fixed time budget.

The narrative is intentionally end to end: start with a familiar OR objective function, compare solver libraries with tracked objective and runtime metrics, promote one configuration to a governed model version, then reuse that same artifact from batch Spark, Python/REST, and SQL.

If you want a longer written walkthrough, start with `blog/inventory-optimization-with-mlflow-on-databricks.md`.

### The example problem

The business story is a distribution-center replenishment plan.

For each SKU, the optimizer decides how many cases to order this week while balancing:

- expected demand
- procurement budget
- storage capacity
- leftover inventory carrying cost
- stockout penalty

This keeps the math approachable while still feeling like an actual inventory optimization workflow instead of a generic knapsack.

### What gets compared

- `OR-Tools CP-SAT`
- `SciPy milp` (HiGHS-backed mixed integer programming)
- `NVIDIA cuOpt` in the serverless GPU companion notebook

The notebook tracks:

- average objective value
- objective subcomponents: gross margin reward, holding cost penalty, and stockout penalty cost
- average, min, p50, p95, and max solve time
- feasible ratio
- optimal ratio
- average fill rate
- shortage rate
- budget utilization
- storage utilization

The champion rule is intentionally simple and reader-friendly: maximize feasible ratio, then maximize fill rate, then maximize objective value, then minimize solve time.

The small benchmark uses the same SKU counts in the CPU and GPU notebooks by default: `18,36,54,72`. That keeps the introductory comparison focused on solver behavior instead of accidentally comparing different problem sizes.

### Why MLflow fits

Inventory optimization experiments still create the same operational questions as ML experiments:

- Which solver configuration is best?
- Which run produced the best service level and economics?
- Which version should we promote?
- How do we serve the recommendation logic reliably?

MLflow handles the experiment tracking, artifact storage, registration, and custom inference packaging using [Models From Code](https://mlflow.org/docs/latest/ml/model/models-from-code/).

### OR-ops pattern

The important operational choice is not whether the model accepts a row with arrays or a grouped table. Those are two representations of the same planning scenario:

- `inventory_sku_inputs` stores normalized SKU-level inputs for ETL, BI, audit, and Spark grouped batch optimization.
- `inventory_scenario_requests` stores one replayable request row per scenario with arrays that match the MLflow model signature.
- `inventory_optimization_scenario_results` stores one output row per solved scenario.
- `inventory_optimization_recommendations` stores SKU-level recommendations from the Champion model.

The repo keeps one MLflow model per runtime family. CPU solvers share `inventory_optimization`; cuOpt uses `inventory_optimization_cuopt` because the dependencies and GPU serving compute are different. Spark batch scoring, Model Serving, and SQL `ai_query` are access paths around those same registered models, not separate model artifacts.

The optional large benchmark is deliberately separate from Champion promotion. It generates one larger scenario, defaults to `2500` SKUs and a `600` second time limit, logs CPU and GPU runs to a large-scale MLflow experiment, and appends comparable outputs to:

- `inventory_large_benchmark_sku_inputs`
- `inventory_large_benchmark_results`
- `inventory_large_benchmark_recommendations`

Use the large benchmark to answer “which runtime finds the best feasible plan under this SLA?” rather than “which small-demo solver should become the governed Champion model?”

The optional network benchmark is a second, more GPU-friendly formulation. It generates a sparse two-echelon distribution network with source-to-DC flows, DC-to-store flows, product/store demand, source capacity, DC throughput, and shortage penalties. It logs CPU `SciPy linprog` and GPU `cuOpt PDLP` runs to a separate experiment and writes comparable outputs to:

- `inventory_network_benchmark_inputs`
- `inventory_network_benchmark_lanes`
- `inventory_network_benchmark_results`
- `inventory_network_benchmark_flows`

Use the network benchmark to test whether cuOpt looks better on a large sparse LP instead of the smaller replenishment MILP.

### Repo layout

- `notebooks/inventory_optimization_crash_course.py` is the main tutorial notebook.
- `notebooks/model_code/inventory_optimizer_model_template.py` is the checked-in MLflow Models From Code template.
- `notebooks/model_code/cuopt_inventory_subprocess.py` isolates cuOpt solver execution for GPU notebook and serving runs.
- `blog/inventory-optimization-with-mlflow-on-databricks.md` gives a longer written walkthrough of the notebook flow.
- `resources/` contains the Databricks bundle resources for registered models and serverless notebook jobs.
- `scripts/deploy_databricks.py` deploys the bundle and runs the notebook job.
- `databricks.yml` defines the Azure target and notebook job variables.

### Run it on Databricks

The project is configured for the `azure` Databricks CLI profile and uses:

- a serverless notebook job
- Unity Catalog for model registration
- Databricks Model Serving for the optional endpoint deployment
- an optional AI Runtime / serverless GPU job for the cuOpt companion workflow

Deploy and run the notebook workflow:

```bash
python scripts/deploy_databricks.py \
  --profile azure \
  --target azure \
  --catalog demos \
  --schema default \
  --model-name inventory_optimization \
  --endpoint-name inventory-optimizer-endpoint \
  --deploy-endpoint true
```

That helper does two things:

1. `databricks bundle deploy` to push the notebook, registered model resources, and serverless job.
2. `databricks bundle run inventory_optimization_crash_course` to execute the notebook with the selected parameters.

The notebook itself performs the benchmark, registers the champion model version, and optionally creates or updates the serving endpoint.

To run the cuOpt companion on serverless GPU compute:

```bash
python scripts/deploy_databricks.py \
  --profile azure \
  --target azure \
  --catalog demos \
  --schema default \
  --resource-key inventory_optimization_cuopt_gpu \
  --gpu-model-name inventory_optimization_cuopt \
  --gpu-endpoint-name inventory-optimizer-cuopt-gpu-endpoint \
  --gpu-hardware-accelerator GPU_1xA10 \
  --gpu-environment-version 4 \
  --gpu-serving-workload-type GPU_SMALL \
  --deploy-endpoint true
```

The GPU notebook benchmarks only cuOpt configurations, then promotes the best cuOpt run to a separate GPU-oriented registered model and serving endpoint. It follows the working routing accelerator pattern by pinning `cuopt-cu12==25.8.0` with `nvidia-nccl-cu12==2.26.2`, then preloading NCCL before importing cuOpt. cuOpt solves run through `notebooks/model_code/cuopt_inventory_subprocess.py` so a native library abort becomes a readable notebook or serving error instead of a dead kernel.

To run the one-scenario large CPU/GPU benchmark without deploying endpoints:

```bash
python scripts/deploy_databricks.py \
  --profile azure \
  --target azure \
  --catalog demos \
  --schema default \
  --resource-key inventory_optimization_large_benchmark \
  --deploy-endpoint false \
  --large-sku-count 2500 \
  --large-time-limit-s 600 \
  --large-benchmark-id inventory_large_2500
```

To run the large sparse network CPU/GPU benchmark without deploying endpoints:

```bash
python scripts/deploy_databricks.py \
  --profile azure \
  --target azure \
  --catalog demos \
  --schema default \
  --resource-key inventory_optimization_network_benchmark \
  --deploy-endpoint false \
  --network-product-count 80 \
  --network-source-count 12 \
  --network-dc-count 80 \
  --network-store-count 250 \
  --network-sources-per-dc 4 \
  --network-dcs-per-store 4 \
  --network-time-limit-s 600 \
  --network-benchmark-id network_80p_80dc_250stores
```

Both notebooks include two inference paths after deployment:

- Spark grouped batch optimization for scheduled OR jobs that write Delta output tables.
- Python/REST for applications and what-if workflows.
- SQL `ai_query` for analysts and endpoint-backed SQL workflows.

For large CPU batch workloads, prefer the Spark `applyInPandas` path shown in the CPU notebook. For cuOpt batch workloads, prefer GPU Model Serving for endpoint-backed use cases or persistent GPU actors/Ray when you need high-throughput in-process solving.

### Open the notebook directly

If you want the most direct tutorial experience, open `notebooks/inventory_optimization_crash_course.py` in Databricks and run it cell by cell. The notebook has widgets for:

- `catalog`
- `schema`
- `experiment_name`
- `model_name`
- `endpoint_name`
- `scenario_count`
- `small_sku_counts`
- `seed`
- `deploy_endpoint`
- `run_large_benchmark`
- `large_sku_count`
- `large_time_limit_s`
- `large_experiment_name`
- `large_benchmark_id`
- `run_network_benchmark`
- `network_product_count`
- `network_source_count`
- `network_dc_count`
- `network_store_count`
- `network_sources_per_dc`
- `network_dcs_per_store`
- `network_time_limit_s`
- `network_experiment_name`
- `network_benchmark_id`

The cuOpt companion adds:

- `gpu_serving_workload_type`

### Suggested blog framing

- Start from the idea that OR experiments deserve the same lifecycle discipline as ML experiments.
- Use weekly inventory replenishment as the motivating supply-chain use case.
- Show how solver settings change fill rate, cost efficiency, and runtime.
- Use MLflow metrics and artifacts as the experiment record, not screenshots or handwritten notes.
- End with the champion solver packaged as an MLflow model, operationalized through Spark batch output tables, optionally deployed on Databricks serverless infrastructure, and queried from Python and SQL.
