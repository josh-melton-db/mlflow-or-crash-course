# Databricks notebook source
# MAGIC %md
# MAGIC # GPU OR-Ops with MLflow + NVIDIA cuOpt
# MAGIC
# MAGIC This companion notebook applies the same OR-ops lifecycle as the CPU walkthrough, but runs the GPU-backed NVIDIA cuOpt solver:
# MAGIC
# MAGIC 1. Reuse the same scenario-level replenishment contract so cuOpt runs land in the same MLflow experiment as the CPU runs.
# MAGIC 2. Benchmark cuOpt configurations against the same scenarios used by the CPU notebook.
# MAGIC 3. Promote the best cuOpt configuration as a separate Unity Catalog model so the GPU and CPU paths each maintain their own Champion alias.
# MAGIC 4. Optionally run a separate large-scale CPU vs GPU benchmark on a sparse distribution-network LP, logged into its own MLflow experiment alongside the CPU notebook.
# MAGIC 5. Pick the right GPU OR-ops access pattern, then deploy the cuOpt champion to GPU Model Serving for interactive, endpoint-backed batch, or `ai_query` access.
# MAGIC
# MAGIC The notebook is intended for Databricks AI Runtime / serverless GPU jobs.

# COMMAND ----------

# MAGIC %pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12==25.8.0 nvidia-nccl-cu12==2.26.2
# MAGIC %pip install "mlflow[databricks]==3.11.1" databricks-sdk==0.103.0 pydantic==2.10.6 typing_extensions==4.15.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "demos")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("experiment_name", "")
dbutils.widgets.text("model_name", "inventory_optimization_cuopt")
dbutils.widgets.text("endpoint_name", "inventory-optimizer-cuopt-gpu-endpoint")
dbutils.widgets.text("scenario_count", "6")
dbutils.widgets.text("small_sku_counts", "18,36,54,72")
dbutils.widgets.text("seed", "7")
dbutils.widgets.dropdown("deploy_endpoint", "true", ["true", "false"])
dbutils.widgets.dropdown("gpu_serving_workload_type", "GPU_SMALL", ["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE", "MULTIGPU_MEDIUM"])
dbutils.widgets.dropdown("run_large_benchmark", "false", ["true", "false"])
dbutils.widgets.text("large_product_count", "80")
dbutils.widgets.text("large_source_count", "12")
dbutils.widgets.text("large_dc_count", "80")
dbutils.widgets.text("large_store_count", "250")
dbutils.widgets.text("large_sources_per_dc", "4")
dbutils.widgets.text("large_dcs_per_store", "4")
dbutils.widgets.text("large_time_limit_s", "600")
dbutils.widgets.text("large_experiment_name", "")
dbutils.widgets.text("large_benchmark_id", "")

# COMMAND ----------

import importlib.metadata as metadata
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep

# Prefer packages installed by the `%pip` cell over runtime-bundled packages.
for package_path in list(sys.path):
    if package_path.startswith("/local_disk0/.ephemeral_nfs/envs/") and package_path.endswith("site-packages"):
        sys.path.remove(package_path)
        sys.path.insert(0, package_path)
sys.modules.pop("typing_extensions", None)

import mlflow
import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceConflict, ResourceDoesNotExist
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
    ServingModelWorkloadType,
    TrafficConfig,
)
from mlflow import MlflowClient

# COMMAND ----------


def parse_sku_counts(raw_value: str, default_counts: list[int]) -> list[int]:
    parsed = [int(part.strip()) for part in raw_value.replace(";", ",").split(",") if part.strip()]
    return parsed or default_counts


catalog = dbutils.widgets.get("catalog").strip() or "demos"
schema = dbutils.widgets.get("schema").strip() or "default"
model_name = dbutils.widgets.get("model_name").strip() or "inventory_optimization_cuopt"
endpoint_name = dbutils.widgets.get("endpoint_name").strip() or "inventory-optimizer-cuopt-gpu-endpoint"
scenario_count = max(3, int(dbutils.widgets.get("scenario_count") or "6"))
small_sku_counts = parse_sku_counts(dbutils.widgets.get("small_sku_counts"), [18, 36, 54, 72])
seed = int(dbutils.widgets.get("seed") or "7")
deploy_endpoint = dbutils.widgets.get("deploy_endpoint").strip().lower() == "true"
gpu_serving_workload_type = dbutils.widgets.get("gpu_serving_workload_type").strip() or "GPU_SMALL"
run_large_benchmark = dbutils.widgets.get("run_large_benchmark").strip().lower() == "true"
large_product_count = int(dbutils.widgets.get("large_product_count") or "80")
large_source_count = int(dbutils.widgets.get("large_source_count") or "12")
large_dc_count = int(dbutils.widgets.get("large_dc_count") or "80")
large_store_count = int(dbutils.widgets.get("large_store_count") or "250")
large_sources_per_dc = int(dbutils.widgets.get("large_sources_per_dc") or "4")
large_dcs_per_store = int(dbutils.widgets.get("large_dcs_per_store") or "4")
large_time_limit_s = float(dbutils.widgets.get("large_time_limit_s") or "600")

registered_model_name = f"{catalog}.{schema}.{model_name}"
current_user = spark.sql("SELECT current_user()").first()[0]
experiment_name = dbutils.widgets.get("experiment_name").strip() or f"/Users/{current_user}/inventory-optimization-crash-course"
large_experiment_name = (
    dbutils.widgets.get("large_experiment_name").strip()
    or f"/Users/{current_user}/inventory-optimization-large-scale"
)
large_benchmark_id = (
    dbutils.widgets.get("large_benchmark_id").strip()
    or f"network_seed{seed}_{large_product_count}p_{large_dc_count}dc_{large_store_count}stores"
)

sku_table_name = f"{catalog}.{schema}.inventory_sku_inputs"
request_table_name = f"{catalog}.{schema}.inventory_scenario_requests"
large_input_table_name = f"{catalog}.{schema}.inventory_large_benchmark_inputs"
large_lane_table_name = f"{catalog}.{schema}.inventory_large_benchmark_lanes"
large_result_table_name = f"{catalog}.{schema}.inventory_large_benchmark_results"
large_flow_table_name = f"{catalog}.{schema}.inventory_large_benchmark_flows"

notebook_workspace_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
workspace_notebook_path = Path(notebook_workspace_path)
if not notebook_workspace_path.startswith("/Workspace/"):
    workspace_notebook_path = Path("/Workspace") / notebook_workspace_path.lstrip("/")
repo_root = workspace_notebook_path.parent.parent
model_template_source = repo_root / "notebooks" / "model_code" / "inventory_optimizer_model_template.py"
if not model_template_source.exists():
    raise FileNotFoundError(f"Model template not found at {model_template_source}")
cuopt_subprocess_source = repo_root / "notebooks" / "model_code" / "cuopt_inventory_subprocess.py"
if not cuopt_subprocess_source.exists():
    raise FileNotFoundError(f"cuOpt subprocess helper not found at {cuopt_subprocess_source}")

workspace = WorkspaceClient()
SERVING_DEPLOYMENT_TIMEOUT = timedelta(minutes=45)

spark.sql(
    f"""
    CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`
    COMMENT 'Inventory optimization crash course assets'
    """
)

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

try:
    gpu_check = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    gpu_summary = gpu_check.stdout.strip() if gpu_check.returncode == 0 else gpu_check.stderr.strip()
except FileNotFoundError:
    gpu_summary = "nvidia-smi was not found; attach this notebook to serverless GPU compute before running."

run_context = {
    "catalog": catalog,
    "schema": schema,
    "experiment_name": experiment_name,
    "model_name": model_name,
    "registered_model_name": registered_model_name,
    "endpoint_name": endpoint_name,
    "large_benchmark_id": large_benchmark_id,
    "large_experiment_name": large_experiment_name,
    "large_input_table_name": large_input_table_name,
    "large_lane_table_name": large_lane_table_name,
    "large_result_table_name": large_result_table_name,
    "large_flow_table_name": large_flow_table_name,
    "large_product_count": large_product_count,
    "large_dc_count": large_dc_count,
    "large_store_count": large_store_count,
    "large_time_limit_s": large_time_limit_s,
    "request_table_name": request_table_name,
    "run_large_benchmark": run_large_benchmark,
    "scenario_count": scenario_count,
    "seed": seed,
    "small_sku_counts": small_sku_counts,
    "sku_table_name": sku_table_name,
    "deploy_endpoint": deploy_endpoint,
    "gpu_serving_workload_type": gpu_serving_workload_type,
    "cuopt_version": metadata.version("cuopt-cu12"),
    "gpu": gpu_summary,
}
print(json.dumps(run_context, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Use the same replenishment formulation
# MAGIC
# MAGIC cuOpt solves the same mixed-integer model as the CPU notebook:
# MAGIC
# MAGIC - integer order, sell, ending inventory, and shortage variables per SKU
# MAGIC - demand balance and inventory balance constraints per SKU
# MAGIC - shared budget and storage capacity constraints
# MAGIC - maximize contribution margin minus holding and stockout penalties
# MAGIC
# MAGIC Within one MLflow experiment, many users, agents, and solver libraries can each log a run against the same business problem. Keeping the OR contract identical across notebooks gives that experiment a uniform scoring surface, so the team can promote a single Champion model under governance no matter which runtime produced it.

# COMMAND ----------

CATEGORY_POOL = [
    "ambient_snacks",
    "beverages",
    "canned_goods",
    "cereal",
    "cleaning",
    "condiments",
    "dairy",
    "frozen_meals",
    "personal_care",
    "produce",
]


def generate_inventory_scenario(scenario_id: str, sku_count: int, scenario_seed: int) -> tuple[pd.DataFrame, int, int]:
    rng = np.random.default_rng(scenario_seed)
    sku_df = pd.DataFrame(
        {
            "sku_sequence": np.arange(1, sku_count + 1, dtype=int),
            "sku_id": [f"{scenario_id.upper()}_SKU_{index + 1:03d}" for index in range(sku_count)],
            "category": rng.choice(CATEGORY_POOL, size=sku_count, replace=True),
            "on_hand_cases": rng.integers(4, 36, size=sku_count),
            "forecast_cases": rng.integers(14, 90, size=sku_count),
            "unit_cost": rng.integers(8, 42, size=sku_count),
            "unit_margin": rng.integers(5, 18, size=sku_count),
            "holding_cost": rng.integers(1, 4, size=sku_count),
            "stockout_penalty": rng.integers(7, 21, size=sku_count),
            "storage_units_per_case": rng.integers(1, 5, size=sku_count),
            "max_order_cases": rng.integers(12, 65, size=sku_count),
        }
    )
    budget = int((sku_df["unit_cost"] * sku_df["max_order_cases"]).sum() * rng.uniform(0.36, 0.52))
    storage_capacity = int(
        (sku_df["storage_units_per_case"] * sku_df["max_order_cases"]).sum() * rng.uniform(0.38, 0.54)
    )
    return sku_df, budget, storage_capacity


def scenario_to_request_record(
    scenario_id: str,
    sku_df: pd.DataFrame,
    budget: int,
    storage_capacity: int,
) -> dict[str, object]:
    ordered = sku_df.sort_values("sku_sequence").reset_index(drop=True)
    return {
        "scenario_id": scenario_id,
        "sku_ids": ordered["sku_id"].tolist(),
        "on_hand": ordered["on_hand_cases"].astype(int).tolist(),
        "forecast": ordered["forecast_cases"].astype(int).tolist(),
        "unit_cost": ordered["unit_cost"].astype(int).tolist(),
        "unit_margin": ordered["unit_margin"].astype(int).tolist(),
        "holding_cost": ordered["holding_cost"].astype(int).tolist(),
        "stockout_penalty": ordered["stockout_penalty"].astype(int).tolist(),
        "storage_units": ordered["storage_units_per_case"].astype(int).tolist(),
        "max_order": ordered["max_order_cases"].astype(int).tolist(),
        "budget": int(budget),
        "storage_capacity": int(storage_capacity),
    }


def scenario_to_sku_table(
    scenario_id: str,
    sku_df: pd.DataFrame,
    budget: int,
    storage_capacity: int,
) -> pd.DataFrame:
    return (
        sku_df.copy()
        .assign(
            scenario_id=scenario_id,
            budget=int(budget),
            storage_capacity=int(storage_capacity),
        )[
            [
                "scenario_id",
                "sku_sequence",
                "sku_id",
                "category",
                "on_hand_cases",
                "forecast_cases",
                "unit_cost",
                "unit_margin",
                "holding_cost",
                "stockout_penalty",
                "storage_units_per_case",
                "max_order_cases",
                "budget",
                "storage_capacity",
            ]
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Persist the shared OR-ops input contract
# MAGIC
# MAGIC The GPU path uses the same data contract as the CPU notebook. The SKU-long table is the operational source shape; the request table stores one scenario-level row with arrays that can be replayed through local validation, GPU Model Serving, or SQL `ai_query`.

# COMMAND ----------

example_scenario_id = "retail_dc_week_demo"
example_sku_df, example_budget, example_storage_capacity = generate_inventory_scenario(
    example_scenario_id,
    sku_count=12,
    scenario_seed=seed,
)
example_request = scenario_to_request_record(
    example_scenario_id,
    example_sku_df,
    example_budget,
    example_storage_capacity,
)
example_sku_table = scenario_to_sku_table(
    example_scenario_id,
    example_sku_df,
    example_budget,
    example_storage_capacity,
)

spark.createDataFrame(example_sku_table).write.mode("overwrite").saveAsTable(sku_table_name)
spark.createDataFrame([example_request]).write.mode("overwrite").saveAsTable(request_table_name)
spark.sql(
    f"COMMENT ON TABLE {sku_table_name} IS 'Normalized SKU-level replenishment inputs for OR-ops optimization examples.'"
)
spark.sql(
    f"COMMENT ON TABLE {request_table_name} IS 'Scenario-level request snapshots that match the registered optimizer model signature.'"
)

display(spark.table(sku_table_name).orderBy("scenario_id", "sku_sequence"))
display(spark.table(request_table_name))

# COMMAND ----------

def summarize_solution(
    *,
    scenario_id: str,
    sku_df: pd.DataFrame,
    budget: int,
    storage_capacity: int,
    library: str,
    config_name: str,
    status: str,
    solve_time_ms: float,
    order_cases: np.ndarray,
    sell_cases: np.ndarray,
    ending_inventory: np.ndarray,
    shortage_cases: np.ndarray,
    is_feasible: bool,
    is_optimal: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    result_df = sku_df.copy()
    result_df["order_cases"] = order_cases.astype(int)
    result_df["sell_cases"] = sell_cases.astype(int)
    result_df["ending_inventory_cases"] = ending_inventory.astype(int)
    result_df["shortage_cases"] = shortage_cases.astype(int)
    result_df["order_spend"] = result_df["order_cases"] * result_df["unit_cost"]
    result_df["storage_used"] = result_df["order_cases"] * result_df["storage_units_per_case"]
    result_df["gross_margin_reward"] = result_df["sell_cases"] * result_df["unit_margin"]
    result_df["holding_cost_penalty"] = result_df["ending_inventory_cases"] * result_df["holding_cost"]
    result_df["stockout_penalty_cost"] = result_df["shortage_cases"] * result_df["stockout_penalty"]
    result_df["objective_component"] = (
        result_df["gross_margin_reward"]
        - result_df["holding_cost_penalty"]
        - result_df["stockout_penalty_cost"]
    )

    total_demand = max(int(result_df["forecast_cases"].sum()), 1)
    total_sold = int(result_df["sell_cases"].sum())
    total_shortage_cases = int(result_df["shortage_cases"].sum())
    total_order_spend = float(result_df["order_spend"].sum())
    total_storage_used = float(result_df["storage_used"].sum())
    objective_value = float(result_df["objective_component"].sum())

    summary = {
        "scenario_id": scenario_id,
        "sku_count": int(len(result_df)),
        "library": library,
        "config_name": config_name,
        "status": status,
        "solve_time_ms": float(solve_time_ms),
        "objective_value": objective_value,
        "gross_margin_reward": float(result_df["gross_margin_reward"].sum()),
        "holding_cost_penalty": float(result_df["holding_cost_penalty"].sum()),
        "stockout_penalty_cost": float(result_df["stockout_penalty_cost"].sum()),
        "is_feasible": int(is_feasible),
        "is_optimal": int(is_optimal),
        "fill_rate": float(total_sold / total_demand),
        "shortage_rate": float(total_shortage_cases / total_demand),
        "budget_utilization": float(total_order_spend / budget),
        "budget_slack": float(budget - total_order_spend),
        "storage_utilization": float(total_storage_used / storage_capacity),
        "storage_slack": float(storage_capacity - total_storage_used),
        "objective_per_sku": float(objective_value / max(len(result_df), 1)),
        "objective_per_1k_cases": float(objective_value / total_demand * 1000),
        "ordered_sku_count": int((result_df["order_cases"] > 0).sum()),
        "total_order_cases": int(result_df["order_cases"].sum()),
        "total_order_spend": total_order_spend,
        "total_storage_used": total_storage_used,
        "total_shortage_cases": total_shortage_cases,
    }
    return summary, result_df


def run_cuopt_subprocess(payload: dict[str, object], timeout_s: float) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(cuopt_subprocess_source)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=max(120, int(timeout_s) + 120),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "cuOpt failed in an isolated Python subprocess. "
            "This usually means the native CUDA/cuOpt libraries are incompatible with the current GPU runtime. "
            f"Return code: {completed.returncode}\n"
            f"stderr tail:\n{completed.stderr[-4000:]}\n"
            f"stdout tail:\n{completed.stdout[-4000:]}"
        )

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("CUOPT_RESULT_JSON:"):
            return json.loads(line.removeprefix("CUOPT_RESULT_JSON:"))
    raise RuntimeError(f"cuOpt subprocess finished without a parseable result. stdout:\n{completed.stdout[-4000:]}")


def solve_with_cuopt(
    scenario_id: str,
    sku_df: pd.DataFrame,
    budget: int,
    storage_capacity: int,
    *,
    config_name: str,
    time_limit_s: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    payload = {
        "scenario_id": scenario_id,
        "sku_rows": sku_df.to_dict(orient="records"),
        "budget": int(budget),
        "storage_capacity": int(storage_capacity),
        "time_limit_s": float(time_limit_s),
    }
    result = run_cuopt_subprocess(payload, timeout_s=time_limit_s)

    return summarize_solution(
        scenario_id=scenario_id,
        sku_df=sku_df,
        budget=budget,
        storage_capacity=storage_capacity,
        library="cuopt_milp",
        config_name=config_name,
        status=str(result["status"]),
        solve_time_ms=float(result["solve_time_ms"]),
        order_cases=np.asarray(result["order_cases"], dtype=int),
        sell_cases=np.asarray(result["sell_cases"], dtype=int),
        ending_inventory=np.asarray(result["ending_inventory"], dtype=int),
        shortage_cases=np.asarray(result["shortage_cases"], dtype=int),
        is_feasible=bool(result["is_feasible"]),
        is_optimal=bool(result["is_optimal"]),
    )

# COMMAND ----------

def benchmark_config(config: dict[str, object], scenarios: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[str, object]]:
    scenario_rows = []
    for scenario in scenarios:
        record, _ = solve_with_cuopt(
            scenario["scenario_id"],
            scenario["sku_df"],
            scenario["budget"],
            scenario["storage_capacity"],
            config_name=config["name"],
            **config["params"],
        )
        scenario_rows.append(record)

    frame = pd.DataFrame(scenario_rows)
    solve_times = frame["solve_time_ms"].astype(float)
    summary = {
        "config_name": config["name"],
        "library": config["library"],
        "avg_objective": float(frame["objective_value"].mean()),
        "total_objective": float(frame["objective_value"].sum()),
        "avg_gross_margin_reward": float(frame["gross_margin_reward"].mean()),
        "avg_holding_cost_penalty": float(frame["holding_cost_penalty"].mean()),
        "avg_stockout_penalty_cost": float(frame["stockout_penalty_cost"].mean()),
        "avg_objective_per_sku": float(frame["objective_per_sku"].mean()),
        "avg_objective_per_1k_cases": float(frame["objective_per_1k_cases"].mean()),
        "avg_solve_time_ms": float(frame["solve_time_ms"].mean()),
        "min_solve_time_ms": float(solve_times.min()),
        "p50_solve_time_ms": float(solve_times.quantile(0.50)),
        "p95_solve_time_ms": float(solve_times.quantile(0.95)),
        "max_solve_time_ms": float(solve_times.max()),
        "feasible_ratio": float(frame["is_feasible"].mean()),
        "optimal_ratio": float(frame["is_optimal"].mean()),
        "avg_fill_rate": float(frame["fill_rate"].mean()),
        "avg_shortage_rate": float(frame["shortage_rate"].mean()),
        "avg_budget_utilization": float(frame["budget_utilization"].mean()),
        "avg_budget_slack": float(frame["budget_slack"].mean()),
        "avg_storage_utilization": float(frame["storage_utilization"].mean()),
        "avg_storage_slack": float(frame["storage_slack"].mean()),
        "avg_ordered_sku_count": float(frame["ordered_sku_count"].mean()),
        "avg_total_shortage_cases": float(frame["total_shortage_cases"].mean()),
        "scenario_count": int(len(frame)),
    }
    for key, value in config["params"].items():
        summary[f"param__{key}"] = value
    return frame, summary


def mlflow_metric_dict(summary: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in summary.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and not key.startswith("param__")
    }


def finite_mlflow_metric_dict(summary: dict[str, object]) -> dict[str, float]:
    return {key: value for key, value in mlflow_metric_dict(summary).items() if np.isfinite(value)}


def records_for_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def generate_distribution_network_scenario(
    *,
    scenario_id: str,
    product_count: int,
    source_count: int,
    dc_count: int,
    store_count: int,
    sources_per_dc: int,
    dcs_per_store: int,
    scenario_seed: int,
    time_limit_s: float,
) -> dict[str, object]:
    rng = np.random.default_rng(scenario_seed)
    product_ids = [f"P{index + 1:04d}" for index in range(product_count)]
    source_ids = [f"SRC{index + 1:03d}" for index in range(source_count)]
    dc_ids = [f"DC{index + 1:03d}" for index in range(dc_count)]
    store_ids = [f"STORE{index + 1:04d}" for index in range(store_count)]

    products = pd.DataFrame(
        {
            "product_id": product_ids,
            "category": rng.choice(CATEGORY_POOL, size=product_count, replace=True),
            "handling_factor": rng.uniform(0.85, 1.35, size=product_count).round(4),
            "shortage_penalty": rng.uniform(150.0, 420.0, size=product_count).round(4),
        }
    )
    stores = pd.DataFrame(
        {
            "store_id": store_ids,
            "region": rng.choice(["north", "south", "east", "west", "central"], size=store_count, replace=True),
            "demand_factor": rng.uniform(0.65, 1.75, size=store_count).round(4),
        }
    )

    base_demand = rng.integers(6, 42, size=(product_count, store_count)).astype(float)
    product_factor = rng.uniform(0.75, 1.4, size=(product_count, 1))
    store_factor = stores["demand_factor"].to_numpy(dtype=float).reshape(1, store_count)
    demand_matrix = np.maximum(1, np.rint(base_demand * product_factor * store_factor)).astype(float)
    demand = pd.DataFrame(
        {
            "product_id": np.repeat(product_ids, store_count),
            "store_id": np.tile(store_ids, product_count),
            "demand_cases": demand_matrix.reshape(-1),
        }
    )
    total_demand = float(demand["demand_cases"].sum())

    source_weights = rng.uniform(0.7, 1.4, size=source_count)
    sources = pd.DataFrame(
        {
            "source_id": source_ids,
            "capacity_cases": np.rint(total_demand * 1.18 * source_weights / source_weights.sum()).astype(float),
        }
    )
    dc_weights = rng.uniform(0.7, 1.5, size=dc_count)
    dcs = pd.DataFrame(
        {
            "dc_id": dc_ids,
            "throughput_capacity_cases": np.rint(total_demand * 1.12 * dc_weights / dc_weights.sum()).astype(float),
        }
    )

    source_dc_records = []
    for dc_id in dc_ids:
        chosen_sources = rng.choice(source_ids, size=min(source_count, sources_per_dc), replace=False)
        for source_id in chosen_sources:
            source_dc_records.append(
                {
                    "lane_id": f"{source_id}_{dc_id}",
                    "source_id": source_id,
                    "dc_id": dc_id,
                    "store_id": None,
                    "cost_per_case": round(float(rng.uniform(0.65, 4.75)), 4),
                }
            )
    source_dc_lanes = pd.DataFrame(source_dc_records).drop_duplicates(["source_id", "dc_id"]).reset_index(drop=True)

    dc_store_records = []
    for store_id in store_ids:
        chosen_dcs = rng.choice(dc_ids, size=min(dc_count, dcs_per_store), replace=False)
        for dc_id in chosen_dcs:
            dc_store_records.append(
                {
                    "lane_id": f"{dc_id}_{store_id}",
                    "source_id": None,
                    "dc_id": dc_id,
                    "store_id": store_id,
                    "cost_per_case": round(float(rng.uniform(1.25, 8.5)), 4),
                }
            )
    dc_store_lanes = pd.DataFrame(dc_store_records).drop_duplicates(["dc_id", "store_id"]).reset_index(drop=True)

    payload = {
        "problem_type": "distribution_network_lp",
        "scenario_id": scenario_id,
        "time_limit_s": float(time_limit_s),
        "products": records_for_payload(products),
        "sources": records_for_payload(sources),
        "dcs": records_for_payload(dcs),
        "stores": records_for_payload(stores[["store_id"]]),
        "demand": records_for_payload(demand),
        "source_dc_lanes": records_for_payload(source_dc_lanes[["source_id", "dc_id", "cost_per_case"]]),
        "dc_store_lanes": records_for_payload(dc_store_lanes[["dc_id", "store_id", "cost_per_case"]]),
    }
    return {
        "payload": payload,
        "products": products,
        "sources": sources,
        "dcs": dcs,
        "stores": stores,
        "demand": demand,
        "source_dc_lanes": source_dc_lanes,
        "dc_store_lanes": dc_store_lanes,
        "total_demand": total_demand,
    }


def network_input_frame(network: dict[str, object], benchmark_id: str, scenario_seed: int) -> pd.DataFrame:
    products = network["products"].assign(entity_type="product")
    sources = network["sources"].assign(entity_type="source")
    dcs = network["dcs"].assign(entity_type="dc")
    stores = network["stores"].assign(entity_type="store")
    demand = network["demand"].assign(entity_type="demand")
    frame = pd.concat([products, sources, dcs, stores, demand], ignore_index=True, sort=False)
    return frame.assign(
        benchmark_id=benchmark_id,
        benchmark_mode="large_scale_cpu_gpu",
        scenario_id=network["payload"]["scenario_id"],
        scenario_seed=int(scenario_seed),
    )


def network_lane_frame(network: dict[str, object], benchmark_id: str, scenario_seed: int) -> pd.DataFrame:
    source_dc = network["source_dc_lanes"].assign(lane_type="source_to_dc")
    dc_store = network["dc_store_lanes"].assign(lane_type="dc_to_store")
    return pd.concat([source_dc, dc_store], ignore_index=True, sort=False).assign(
        benchmark_id=benchmark_id,
        benchmark_mode="large_scale_cpu_gpu",
        scenario_id=network["payload"]["scenario_id"],
        scenario_seed=int(scenario_seed),
    )


def solve_network_with_cuopt(
    network: dict[str, object],
    *,
    config_name: str,
    time_limit_s: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    result = run_cuopt_subprocess(network["payload"], timeout_s=time_limit_s)
    flow_frame = pd.DataFrame(result.pop("flows", []))
    summary = {
        "scenario_id": network["payload"]["scenario_id"],
        "library": "cuopt_pdlp",
        "config_name": config_name,
        "problem_type": "distribution_network_lp",
        **result,
        "shortage_rate": float(result["shortage_cases"] / result["total_demand"] if result["total_demand"] else 0.0),
        "product_count": int(len(network["products"])),
        "source_count": int(len(network["sources"])),
        "dc_count": int(len(network["dcs"])),
        "store_count": int(len(network["stores"])),
        "source_dc_lane_count": int(len(network["source_dc_lanes"])),
        "dc_store_lane_count": int(len(network["dc_store_lanes"])),
        "variable_count": int(
            len(network["products"])
            * (len(network["source_dc_lanes"]) + len(network["dc_store_lanes"]) + len(network["stores"]))
        ),
        "constraint_count": int(
            len(network["sources"])
            + len(network["dcs"])
            + len(network["products"]) * len(network["dcs"])
            + len(network["products"]) * len(network["stores"])
        ),
        "time_limit_s": float(time_limit_s),
    }
    summary["is_feasible"] = int(bool(summary["is_feasible"]))
    summary["is_optimal"] = int(bool(summary["is_optimal"]))
    return summary, flow_frame


def run_benchmark(
    configs: list[dict[str, object]],
    scenarios: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_rows = []
    scenario_frames = {}
    for config in configs:
        scenario_frame, summary = benchmark_config(config, scenarios)
        summary_rows.append(summary)
        scenario_frames[config["name"]] = scenario_frame

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        by=["feasible_ratio", "avg_fill_rate", "avg_objective", "avg_solve_time_ms"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return summary_frame, scenario_frames


def log_solver_run(config: dict[str, object], scenario_frame: pd.DataFrame, summary: dict[str, object]) -> None:
    with mlflow.start_run(run_name=config["name"], nested=True):
        mlflow.log_param("library", config["library"])
        mlflow.log_param("accelerator", "serverless_gpu")
        for key, value in config["params"].items():
            mlflow.log_param(f"solver__{key}", value)
        mlflow.log_metrics(mlflow_metric_dict(summary))
        mlflow.log_table(scenario_frame, artifact_file=f"benchmark/{config['name']}_scenario_results.json")


def select_champion(summary_frame: pd.DataFrame) -> dict[str, object]:
    if summary_frame.empty:
        raise ValueError("No candidate rows found.")
    champion_row = summary_frame.iloc[0].to_dict()
    champion_row["selection_rule"] = "among cuOpt runs: max feasible_ratio, max avg_fill_rate, max avg_objective, min avg_solve_time_ms"
    return champion_row


def log_and_register_model(
    *,
    champion_config: dict[str, object],
    run_id: str,
    temp_root: Path,
) -> tuple[object, str]:
    model_script_path = render_model_script(champion_config, temp_root / f"{model_name}_model.py")
    model_signature = mlflow.models.infer_signature(
        pd.DataFrame([example_request]),
        pd.DataFrame([example_record]),
    )
    log_model_kwargs = {}
    if champion_config["library"] == "cuopt_milp":
        log_model_kwargs["code_paths"] = [str(cuopt_subprocess_source)]
    model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model=str(model_script_path),
        registered_model_name=registered_model_name,
        signature=model_signature,
        input_example=[example_request],
        pip_requirements=build_model_requirements(champion_config["library"]),
        **log_model_kwargs,
    )
    model_version = resolve_logged_model_version(registered_model_name, run_id)
    return model_info, model_version


def build_model_requirements(library: str) -> list[str]:
    if library != "cuopt_milp":
        raise ValueError(f"Unsupported GPU notebook library: {library}")
    requirements = [
        f"mlflow[databricks]=={metadata.version('mlflow')}",
        f"pydantic=={metadata.version('pydantic')}",
        f"typing_extensions=={metadata.version('typing_extensions')}",
        "--extra-index-url https://pypi.nvidia.com",
        "cuopt-cu12==25.8.0",
        "nvidia-nccl-cu12==2.26.2",
    ]
    return requirements


def render_model_script(champion_config: dict[str, object], output_path: Path) -> Path:
    rendered_script = (
        model_template_source.read_text(encoding="utf-8")
        .replace("__MODEL_LIBRARY__", champion_config["library"])
        .replace("__MODEL_CONFIG_NAME__", champion_config["name"])
        .replace("__MODEL_PARAMS_JSON__", json.dumps(champion_config["params"], sort_keys=True))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered_script, encoding="utf-8")
    return output_path


def resolve_logged_model_version(registered_model_name: str, run_id: str) -> str:
    client = MlflowClient()
    matching_versions = [
        version
        for version in client.search_model_versions(f"name='{registered_model_name}'")
        if version.run_id == run_id
    ]
    latest_version = max(matching_versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(registered_model_name, "Champion", latest_version.version)
    return str(latest_version.version)


def wait_for_endpoint_not_updating(endpoint_name: str, timeout: timedelta = SERVING_DEPLOYMENT_TIMEOUT) -> None:
    deadline = perf_counter() + timeout.total_seconds()
    while True:
        endpoint = workspace.serving_endpoints.get(endpoint_name)
        config_update_state = str(getattr(endpoint.state, "config_update", ""))
        if not config_update_state.endswith("IN_PROGRESS"):
            return
        if perf_counter() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for endpoint {endpoint_name} to finish its current config update."
            )
        sleep(20)


def create_or_update_gpu_endpoint(
    endpoint_name: str,
    registered_model_name: str,
    model_version: str,
    workload_type_name: str,
) -> dict[str, str]:
    workload_type = getattr(ServingModelWorkloadType, workload_type_name)
    served_model_name = f"{registered_model_name.split('.')[-1]}-{model_version}"
    served_entities = [
        ServedEntityInput(
            entity_name=registered_model_name,
            entity_version=str(model_version),
            name=served_model_name,
            workload_size="Small",
            workload_type=workload_type,
            scale_to_zero_enabled=True,
        )
    ]
    traffic_config = TrafficConfig(routes=[Route(served_model_name=served_model_name, traffic_percentage=100)])

    try:
        existing_endpoint = workspace.serving_endpoints.get(endpoint_name)
    except ResourceDoesNotExist:
        existing_endpoint = None

    if existing_endpoint is None:
        workspace.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=served_entities,
                traffic_config=traffic_config,
            ),
            timeout=SERVING_DEPLOYMENT_TIMEOUT,
        )
        action = "created"
    else:
        wait_for_endpoint_not_updating(endpoint_name)
        try:
            workspace.serving_endpoints.update_config_and_wait(
                name=endpoint_name,
                served_entities=served_entities,
                traffic_config=traffic_config,
                timeout=SERVING_DEPLOYMENT_TIMEOUT,
            )
        except ResourceConflict:
            wait_for_endpoint_not_updating(endpoint_name)
            workspace.serving_endpoints.update_config_and_wait(
                name=endpoint_name,
                served_entities=served_entities,
                traffic_config=traffic_config,
                timeout=SERVING_DEPLOYMENT_TIMEOUT,
            )
        action = "updated"

    return {
        "action": action,
        "endpoint_name": endpoint_name,
        "registered_model_name": registered_model_name,
        "model_version": str(model_version),
        "served_model_name": served_model_name,
        "workload_type": workload_type_name,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Smoke-test cuOpt on the small example
# MAGIC
# MAGIC Before kicking off the full benchmark, solve the same explainable scenario with cuOpt. This is the GPU equivalent of the small solve in the CPU walkthrough — quick to read, easy to compare against the recommended order plan you already know is sensible.

# COMMAND ----------

example_record, example_solution = solve_with_cuopt(
    example_scenario_id,
    example_sku_df,
    example_budget,
    example_storage_capacity,
    config_name="cuopt_baseline_demo",
    time_limit_s=4.0,
)

display(
    example_solution.loc[
        (example_solution["order_cases"] > 0) | (example_solution["shortage_cases"] > 0),
        [
            "sku_id",
            "category",
            "forecast_cases",
            "on_hand_cases",
            "order_cases",
            "sell_cases",
            "ending_inventory_cases",
            "shortage_cases",
            "order_spend",
            "storage_used",
            "gross_margin_reward",
            "holding_cost_penalty",
            "stockout_penalty_cost",
            "objective_component",
        ],
    ]
    .sort_values(["order_cases", "shortage_cases"], ascending=False)
    .reset_index(drop=True)
)
display(pd.DataFrame([example_record]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define a cuOpt benchmark sweep
# MAGIC
# MAGIC This sweep evaluates cuOpt configurations against the same scenarios the CPU notebook uses, so the runs land in the same MLflow experiment and can be compared side by side. CPU solvers stay in the CPU notebook because their packages and runtime fit standard serverless compute, and cuOpt stays here because it needs GPU compute and CUDA libraries.

# COMMAND ----------

sku_counts = np.asarray(small_sku_counts, dtype=int)
benchmark_scenarios = []
for index, sku_count in enumerate(sku_counts, start=1):
    scenario_id = f"gpu_week_{index:02d}_{sku_count}skus"
    sku_df, budget, storage_capacity = generate_inventory_scenario(scenario_id, int(sku_count), seed + index)
    benchmark_scenarios.append(
        {
            "scenario_id": scenario_id,
            "sku_df": sku_df,
            "budget": budget,
            "storage_capacity": storage_capacity,
        }
    )

solver_configs = [
    {
        "name": "cuopt_gpu_fast",
        "library": "cuopt_milp",
        "params": {"time_limit_s": 4.0},
    },
    {
        "name": "cuopt_gpu_longer",
        "library": "cuopt_milp",
        "params": {"time_limit_s": 8.0},
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run the shared MLflow experiment and register the cuOpt champion
# MAGIC
# MAGIC The cuOpt runs land in the same MLflow experiment as the CPU runs from the main notebook. That gives the team one place to compare every attempt — different libraries, different parameters, different users, different agents — against the same scenarios.
# MAGIC
# MAGIC The registered model is its own Unity Catalog model (`inventory_optimization_cuopt`) because cuOpt needs GPU serving compute and an NVIDIA-specific dependency stack. The CPU and GPU walkthroughs each maintain their own Champion alias, so promoting one does not silently overwrite the other.

# COMMAND ----------

run_name = f"inventory_cuopt_gpu_benchmark_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"
mlflow.set_experiment(experiment_name)

with TemporaryDirectory() as temp_dir:
    temp_root = Path(temp_dir)
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(
            {
                "problem_type": "inventory_replenishment",
                "benchmark_mode": "cuopt_gpu_companion",
                "scenario_count": scenario_count,
                "seed": seed,
                "catalog": catalog,
                "schema": schema,
                "model_name": model_name,
                "registered_model_name": registered_model_name,
                "deploy_endpoint": int(deploy_endpoint),
                "gpu_serving_workload_type": gpu_serving_workload_type,
                "cuopt_version": metadata.version("cuopt-cu12"),
            }
        )
        mlflow.set_tags(
            {
                "accelerator": "serverless_gpu",
                "partner_solver": "nvidia_cuopt",
                "promotion_scope": "best_cuopt_configuration",
            }
        )
        mlflow.log_dict(example_request, "artifacts/input_example.json")
        mlflow.log_table(example_solution, artifact_file="artifacts/cuopt_example_solution.json")

        summary_frame, scenario_frames = run_benchmark(solver_configs, benchmark_scenarios)
        for config in solver_configs:
            summary = summary_frame[summary_frame["config_name"] == config["name"]].iloc[0].to_dict()
            log_solver_run(config, scenario_frames[config["name"]], summary)

        cuopt_summary_frame = summary_frame.reset_index(drop=True)
        cuopt_champion_row = select_champion(summary_frame)
        champion_config = next(config for config in solver_configs if config["name"] == cuopt_champion_row["config_name"])

        mlflow.log_table(summary_frame, artifact_file="benchmark/solver_comparison.json")
        mlflow.log_dict(cuopt_champion_row, "benchmark/cuopt_champion.json")
        mlflow.log_metrics(
            {
                f"cuopt_champion_{key}": value
                for key, value in mlflow_metric_dict(cuopt_champion_row).items()
                if key not in {"scenario_count"}
            }
        )
        mlflow.set_tag("cuopt_champion_config_name", cuopt_champion_row["config_name"])

        model_info, model_version = log_and_register_model(
            champion_config=champion_config,
            run_id=active_run.info.run_id,
            temp_root=temp_root,
        )

local_model = mlflow.pyfunc.load_model(model_info.model_uri)
validation_prediction = local_model.predict([example_request])
if isinstance(validation_prediction, list):
    validation_preview = validation_prediction[0] if validation_prediction else None
elif hasattr(validation_prediction, "to_dict"):
    validation_preview = validation_prediction.to_dict(orient="records")[0]
else:
    validation_preview = validation_prediction

experiment_result = {
    "run_id": active_run.info.run_id,
    "experiment_name": experiment_name,
    "cuopt_champion": cuopt_champion_row,
    "cuopt_champion_config": champion_config,
    "registered_model_name": registered_model_name,
    "registered_model_version": model_version,
    "model_uri": model_info.model_uri,
    "validation_prediction": validation_preview,
}
print(json.dumps(experiment_result, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5b. Optional large-scale cuOpt benchmark
# MAGIC
# MAGIC The small replenishment MILP above is intentionally compact — it is the formulation that gets registered, served, and queried throughout the rest of the notebook.
# MAGIC
# MAGIC OR teams running on Databricks usually also want to know: at what problem size does it pay off to switch runtimes? This optional section logs a separate stress experiment on a larger, sparse distribution-network LP and lands those runs in the same MLflow experiment as the CPU companion notebook. Same generated network, same seed, same time budget — both runtimes show up as comparable rows you can sort by solve time and cost. cuOpt PDLP is exercised here because PDLP is the cuOpt solver designed for large sparse LPs.

# COMMAND ----------

large_benchmark_summary = {"enabled": run_large_benchmark}

if run_large_benchmark:
    large_scenario_seed = seed + 10_000
    large_scenario_id = (
        f"{large_benchmark_id}_{large_product_count}p_"
        f"{large_dc_count}dc_{large_store_count}stores"
    )
    network = generate_distribution_network_scenario(
        scenario_id=large_scenario_id,
        product_count=large_product_count,
        source_count=large_source_count,
        dc_count=large_dc_count,
        store_count=large_store_count,
        sources_per_dc=large_sources_per_dc,
        dcs_per_store=large_dcs_per_store,
        scenario_seed=large_scenario_seed,
        time_limit_s=large_time_limit_s,
    )

    spark.createDataFrame(
        network_input_frame(network, large_benchmark_id, large_scenario_seed)
    ).write.mode("append").option("mergeSchema", "true").saveAsTable(large_input_table_name)
    spark.createDataFrame(
        network_lane_frame(network, large_benchmark_id, large_scenario_seed)
    ).write.mode("append").option("mergeSchema", "true").saveAsTable(large_lane_table_name)

    mlflow.set_experiment(large_experiment_name)
    large_run_name = f"large_distribution_cuopt_{large_benchmark_id}"
    with mlflow.start_run(run_name=large_run_name) as large_active_run:
        mlflow.log_params(
            {
                "benchmark_id": large_benchmark_id,
                "benchmark_mode": "large_scale_cpu_gpu",
                "accelerator": "serverless_gpu",
                "partner_solver": "nvidia_cuopt",
                "problem_type": "distribution_network_lp",
                "scenario_id": large_scenario_id,
                "scenario_seed": large_scenario_seed,
                "product_count": large_product_count,
                "source_count": large_source_count,
                "dc_count": large_dc_count,
                "store_count": large_store_count,
                "sources_per_dc": large_sources_per_dc,
                "dcs_per_store": large_dcs_per_store,
                "time_limit_s": large_time_limit_s,
                "catalog": catalog,
                "schema": schema,
            }
        )
        mlflow.set_tags(
            {
                "benchmark_id": large_benchmark_id,
                "benchmark_mode": "large_scale_cpu_gpu",
                "accelerator": "serverless_gpu",
                "benchmark_scope": "large_cpu_gpu_comparison",
                "partner_solver": "nvidia_cuopt",
            }
        )

        large_record, large_flows = solve_network_with_cuopt(
            network,
            config_name="cuopt_network_pdlp",
            time_limit_s=large_time_limit_s,
        )
        large_record = {
            **large_record,
            "benchmark_id": large_benchmark_id,
            "benchmark_mode": "large_scale_cpu_gpu",
            "mlflow_run_id": large_active_run.info.run_id,
            "scenario_seed": large_scenario_seed,
            "accelerator": "serverless_gpu",
        }
        large_flows = large_flows.assign(
            benchmark_id=large_benchmark_id,
            benchmark_mode="large_scale_cpu_gpu",
            mlflow_run_id=large_active_run.info.run_id,
            scenario_id=large_scenario_id,
            library=large_record["library"],
            config_name=large_record["config_name"],
            status=large_record["status"],
            is_feasible=bool(large_record["is_feasible"]),
            is_optimal=bool(large_record["is_optimal"]),
            solve_time_ms=float(large_record["solve_time_ms"]),
            objective_value=float(large_record["objective_value"]),
            fill_rate=float(large_record["fill_rate"]),
            accelerator="serverless_gpu",
        )

        with mlflow.start_run(run_name="cuopt_network_pdlp", nested=True):
            mlflow.log_param("library", large_record["library"])
            mlflow.log_param("accelerator", "serverless_gpu")
            mlflow.log_param("benchmark_id", large_benchmark_id)
            mlflow.log_param("benchmark_mode", "large_scale_cpu_gpu")
            mlflow.log_param("time_limit_s", large_time_limit_s)
            mlflow.log_metrics(finite_mlflow_metric_dict(large_record))
            mlflow.log_table(pd.DataFrame([large_record]), artifact_file="large_benchmark/gpu_summary.json")

    large_results_frame = pd.DataFrame([large_record])
    spark.createDataFrame(large_results_frame).write.mode("append").option("mergeSchema", "true").saveAsTable(
        large_result_table_name
    )
    spark.createDataFrame(large_flows).write.mode("append").option("mergeSchema", "true").saveAsTable(
        large_flow_table_name
    )
    display(large_results_frame)
    large_benchmark_summary = {
        "enabled": True,
        "benchmark_id": large_benchmark_id,
        "experiment_name": large_experiment_name,
        "run_id": large_active_run.info.run_id,
        "product_count": large_product_count,
        "source_count": large_source_count,
        "dc_count": large_dc_count,
        "store_count": large_store_count,
        "source_dc_lane_count": int(len(network["source_dc_lanes"])),
        "dc_store_lane_count": int(len(network["dc_store_lanes"])),
        "variable_count": int(large_record["variable_count"]),
        "constraint_count": int(large_record["constraint_count"]),
        "time_limit_s": large_time_limit_s,
        "result_table_name": large_result_table_name,
        "flow_table_name": large_flow_table_name,
    }

print(json.dumps(large_benchmark_summary, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Choose the right GPU OR-ops access pattern
# MAGIC
# MAGIC The registered cuOpt model takes the same scenario request as the CPU optimizer, but the right access pattern looks different on GPUs because cuOpt depends on CUDA libraries and GPU hardware. Pick one before wiring up downstream code; the rest of this notebook implements the GPU Model Serving path against this same Champion model artifact.
# MAGIC
# MAGIC | Pattern | Best for | GPU note |
# MAGIC | --- | --- | --- |
# MAGIC | GPU Model Serving | Apps, what-if optimization, SQL `ai_query`, and endpoint-backed batch | Recommended default because the serving endpoint owns the GPU runtime and package environment |
# MAGIC | Ray or persistent GPU actors | High-throughput batch solving where each worker can keep a CUDA context warm | Prefer this over launching a fresh cuOpt process per small group |
# MAGIC | Spark `applyInPandas` | CPU solvers only | Even on serverless GPU compute the GPU is attached only to the driver; Spark executors run in standard CPU containers without `nvidia-smi`, CUDA, or `cuopt` installed, so cuOpt UDFs fail at import time. Use one of the other two patterns instead. |
# MAGIC
# MAGIC A small helper script ships with the MLflow model so native cuOpt failures surface as readable Python errors instead of crashing the notebook or serving worker.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Deploy the cuOpt champion to GPU Model Serving
# MAGIC
# MAGIC GPU Model Serving is the interactive OR-ops access path for cuOpt. The same Champion model the team just promoted is exposed behind an HTTP endpoint, so apps and analysts can solve a single scenario on demand without rerunning the benchmark or reloading the artifact.
# MAGIC
# MAGIC The deployment uses the same Databricks Model Serving API as the CPU walkthrough, with one important difference: the served entity sets `workload_type` to a GPU workload such as `GPU_SMALL`. Endpoint creation or update can take 30 minutes or more on a cold path because package installation pulls cuOpt from NVIDIA's Python package index.

# COMMAND ----------

deployment_result = {
    "skipped": True,
    "reason": "Set deploy_endpoint=true to create or update the GPU serving endpoint.",
}

if deploy_endpoint:
    deployment_result = create_or_update_gpu_endpoint(
        endpoint_name,
        experiment_result["registered_model_name"],
        experiment_result["registered_model_version"],
        gpu_serving_workload_type,
    )
    with mlflow.start_run(run_id=experiment_result["run_id"]):
        mlflow.log_dict(deployment_result, "deployment/gpu_endpoint_result.json")

notebook_result = {
    **experiment_result,
    "deployment_result": deployment_result,
    "large_benchmark_result": large_benchmark_summary,
}
print(json.dumps(notebook_result, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Review cuOpt results and the promoted model
# MAGIC
# MAGIC The summary tables compare cuOpt configurations against the same scenarios the CPU notebook used. The promoted model is the best cuOpt row because this notebook governs the GPU access path; the CPU walkthrough governs its own Champion alias under a separate registered model so both runtimes can coexist.

# COMMAND ----------

display(summary_frame)
display(cuopt_summary_frame)
print(json.dumps(notebook_result, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Query the deployed GPU endpoint from Python
# MAGIC
# MAGIC The endpoint request shape is unchanged from the CPU notebook because the MLflow model uses the same scenario-level input contract.

# COMMAND ----------

python_request = spark.table(request_table_name).collect()[0].asDict(recursive=True)

if deploy_endpoint:
    python_endpoint_response = workspace.api_client.do(
        method="POST",
        path=f"/serving-endpoints/{endpoint_name}/invocations",
        body={"dataframe_records": [python_request]},
    )
    print(json.dumps(python_endpoint_response, indent=2, sort_keys=True))
else:
    print(json.dumps({"dataframe_records": [python_request]}, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Query the GPU endpoint from SQL with `ai_query`
# MAGIC
# MAGIC The same promoted cuOpt endpoint can be called from Databricks SQL. This is useful for analyst workflows or endpoint-backed batch calls where the input rows already live in the request snapshot table. For large cuOpt batch workloads, prefer GPU actors or purpose-built GPU jobs over standard CPU Spark grouped execution.

# COMMAND ----------

ai_query_return_type = """
STRUCT<
  scenario_id: STRING,
  library: STRING,
  config_name: STRING,
  status: STRING,
  is_feasible: BOOLEAN,
  is_optimal: BOOLEAN,
  objective_value: DOUBLE,
  gross_margin_reward: DOUBLE,
  holding_cost_penalty: DOUBLE,
  stockout_penalty_cost: DOUBLE,
  fill_rate: DOUBLE,
  shortage_rate: DOUBLE,
  total_order_spend: DOUBLE,
  total_storage_used: DOUBLE,
  budget_slack: DOUBLE,
  storage_slack: DOUBLE,
  ordered_sku_count: BIGINT,
  total_shortage_cases: BIGINT,
  recommendations: ARRAY<STRUCT<
    sku_id: STRING,
    order_cases: BIGINT,
    sell_cases: BIGINT,
    ending_inventory_cases: BIGINT,
    shortage_cases: BIGINT
  >>
>
"""

ai_query_sql = f"""
SELECT
  scenario_id,
  prediction.result.status AS status,
  prediction.result.objective_value AS objective_value,
  prediction.result.gross_margin_reward AS gross_margin_reward,
  prediction.result.holding_cost_penalty AS holding_cost_penalty,
  prediction.result.stockout_penalty_cost AS stockout_penalty_cost,
  prediction.result.fill_rate AS fill_rate,
  prediction.result.shortage_rate AS shortage_rate,
  prediction.result.ordered_sku_count AS ordered_sku_count,
  prediction.result.recommendations AS recommendations,
  prediction.errorMessage AS error_message
FROM (
  SELECT
    scenario_id,
    ai_query(
      endpoint => '{endpoint_name}',
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
      returnType => '{ai_query_return_type}',
      failOnError => false
    ) AS prediction
  FROM {request_table_name}
)
"""

if deploy_endpoint:
    try:
        display(spark.sql(ai_query_sql))
    except Exception as exc:
        print(
            "The ai_query example could not run in this workspace. "
            "Confirm the AI_Query for Custom Models preview is enabled and that the SQL runtime supports custom model endpoints."
        )
        print(str(exc))
        print(ai_query_sql)
else:
    print(ai_query_sql)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Operating cuOpt on Databricks
# MAGIC
# MAGIC A few things to keep in mind once this notebook is part of a real workflow:
# MAGIC
# MAGIC - Run it on AI Runtime / serverless GPU compute. Standard serverless notebook compute will not have GPUs available.
# MAGIC - The cuOpt package comes from `https://pypi.nvidia.com`. Both the notebook install and the logged MLflow model requirements include that extra index, so serving can install the same versions.
# MAGIC - cuOpt solves go through a small helper script (`notebooks/model_code/cuopt_inventory_subprocess.py`) that the MLflow model carries along. If `libcuopt` ever aborts, this surfaces as a readable Python error instead of a dead notebook kernel or serving worker.
# MAGIC - GPU Model Serving needs `workload_type` set on the served entity, for example `GPU_SMALL`.
# MAGIC - The CPU and GPU walkthroughs each register their own Champion model (`inventory_optimization` vs `inventory_optimization_cuopt`) so promoting one runtime never silently overwrites the other.
# MAGIC - For comparing CPU vs GPU at scale, run the optional large-scale benchmark in section 5b on both notebooks. Both CPU and GPU runs land in the same dedicated MLflow experiment and write comparable rows to the same Delta tables.

# COMMAND ----------

final_summary = {
    "experiment_name": experiment_result["experiment_name"],
    "run_id": experiment_result["run_id"],
    "registered_model_name": experiment_result["registered_model_name"],
    "registered_model_version": experiment_result["registered_model_version"],
    "cuopt_champion_config_name": experiment_result["cuopt_champion"]["config_name"],
    "endpoint_name": deployment_result.get("endpoint_name", endpoint_name),
    "endpoint_action": deployment_result.get("action", "skipped"),
    "gpu_serving_workload_type": deployment_result.get("workload_type", gpu_serving_workload_type),
    "large_benchmark_result": large_benchmark_summary,
    "request_table_name": request_table_name,
    "sku_table_name": sku_table_name,
}
print(json.dumps(final_summary, indent=2, sort_keys=True))
