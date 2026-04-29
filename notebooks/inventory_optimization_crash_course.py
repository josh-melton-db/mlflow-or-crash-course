# Databricks notebook source
# MAGIC %md
# MAGIC # OR-Ops with MLflow + Inventory Optimization
# MAGIC
# MAGIC This notebook is a tactical guide to OR-ops on Databricks: the MLOps-style lifecycle for optimization models.
# MAGIC
# MAGIC 1. Define a scenario-level optimization contract for the small replenishment example.
# MAGIC 2. Persist normalized SKU inputs and request snapshots in Unity Catalog.
# MAGIC 3. Benchmark `OR-Tools CP-SAT` against `SciPy milp` with MLflow and register the winning configuration as a governed model version.
# MAGIC 4. Optionally run a separate large-scale CPU vs GPU benchmark on a sparse distribution-network LP, logged into its own MLflow experiment alongside the GPU companion notebook.
# MAGIC 5. Pick the OR-ops access pattern that fits the use case: Spark batch, Model Serving, or SQL `ai_query`.
# MAGIC 6. Reuse the same governed Champion model from each of those access paths.

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]==3.11.1" ortools==9.15.6755 scipy==1.15.3 pandas==2.3.3 numpy==2.2.6 pyarrow==23.0.1 databricks-sdk==0.103.0 pydantic==2.10.6 typing_extensions==4.15.0

# COMMAND ----------

dbutils.widgets.text("catalog", "demos")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("experiment_name", "")
dbutils.widgets.text("model_name", "inventory_optimization")
dbutils.widgets.text("endpoint_name", "inventory-optimizer-endpoint")
dbutils.widgets.text("scenario_count", "6")
dbutils.widgets.text("small_sku_counts", "18,36,54,72")
dbutils.widgets.text("seed", "7")
dbutils.widgets.dropdown("deploy_endpoint", "true", ["true", "false"])
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
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

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
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import EndpointCoreConfigInput, Route, ServedEntityInput, TrafficConfig
from mlflow import MlflowClient
from ortools.sat.python import cp_model
from pyspark.sql.types import BooleanType, DoubleType, LongType, StringType, StructField, StructType
import scipy.sparse as sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# COMMAND ----------


def parse_sku_counts(raw_value: str, default_counts: list[int]) -> list[int]:
    parsed = [int(part.strip()) for part in raw_value.replace(";", ",").split(",") if part.strip()]
    return parsed or default_counts


catalog = dbutils.widgets.get("catalog").strip() or "demos"
schema = dbutils.widgets.get("schema").strip() or "default"
model_name = dbutils.widgets.get("model_name").strip() or "inventory_optimization"
endpoint_name = dbutils.widgets.get("endpoint_name").strip() or "inventory-optimizer-endpoint"
scenario_count = max(3, int(dbutils.widgets.get("scenario_count") or "6"))
small_sku_counts = parse_sku_counts(dbutils.widgets.get("small_sku_counts"), [18, 36, 54, 72])
seed = int(dbutils.widgets.get("seed") or "7")
deploy_endpoint = dbutils.widgets.get("deploy_endpoint").strip().lower() == "true"
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
scenario_result_table_name = f"{catalog}.{schema}.inventory_optimization_scenario_results"
recommendation_table_name = f"{catalog}.{schema}.inventory_optimization_recommendations"
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

workspace = WorkspaceClient()

spark.sql(
    f"""
    CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`
    COMMENT 'Inventory optimization crash course assets'
    """
)

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

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
    "recommendation_table_name": recommendation_table_name,
    "request_table_name": request_table_name,
    "run_large_benchmark": run_large_benchmark,
    "scenario_count": scenario_count,
    "scenario_result_table_name": scenario_result_table_name,
    "seed": seed,
    "small_sku_counts": small_sku_counts,
    "sku_table_name": sku_table_name,
    "deploy_endpoint": deploy_endpoint,
}
print(json.dumps(run_context, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Frame the replenishment decision
# MAGIC
# MAGIC Each row in the example represents one SKU in a weekly distribution-center planning cycle.
# MAGIC
# MAGIC The optimizer decides how many cases to reorder while balancing three business goals:
# MAGIC
# MAGIC - protect service level by fulfilling as much forecast demand as possible
# MAGIC - stay inside the procurement budget and storage capacity
# MAGIC - avoid both excess leftover inventory and costly stockouts

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision variables and trade-offs
# MAGIC
# MAGIC For each SKU, the model tracks four quantities:
# MAGIC
# MAGIC - `order_cases`: how many new cases to buy
# MAGIC - `sell_cases`: how many forecast cases can be fulfilled
# MAGIC - `ending_inventory_cases`: how many cases remain after the week closes
# MAGIC - `shortage_cases`: how many cases of demand remain unmet
# MAGIC
# MAGIC The objective rewards fulfilled demand and penalizes leftover inventory plus shortages. That gives us a compact formulation that feels realistic without adding too much notation.

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
# MAGIC ## 2. Persist the OR-ops input contract
# MAGIC
# MAGIC OR-ops starts by making the optimization contract explicit. The operational source table stays in normal SKU-long form, while the request snapshot stores one replayable row per planning scenario with array fields that match the MLflow model signature.
# MAGIC
# MAGIC - `inventory_sku_inputs`: one row per SKU per scenario, useful for ETL, BI, audit, and grouped Spark batch optimization.
# MAGIC - `inventory_scenario_requests`: one row per scenario, useful for serving, `ai_query`, and exact replay of a solved request.
# MAGIC - `inventory_optimization_scenario_results` and `inventory_optimization_recommendations`: output tables populated after the champion model is registered.
# MAGIC
# MAGIC The small demo scenario is kept first so it can still anchor the explanatory solve and MLflow input example.

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

operational_scenarios = [
    {
        "scenario_id": example_scenario_id,
        "sku_df": example_sku_df,
        "budget": example_budget,
        "storage_capacity": example_storage_capacity,
    }
]
operational_sku_counts = np.linspace(16, 64, num=scenario_count, dtype=int)
for index, sku_count in enumerate(operational_sku_counts, start=1):
    scenario_id = f"ops_week_{index:02d}_{sku_count}skus"
    sku_df, budget, storage_capacity = generate_inventory_scenario(scenario_id, int(sku_count), seed + 100 + index)
    operational_scenarios.append(
        {
            "scenario_id": scenario_id,
            "sku_df": sku_df,
            "budget": budget,
            "storage_capacity": storage_capacity,
        }
    )

operational_sku_table = pd.concat(
    [
        scenario_to_sku_table(
            scenario["scenario_id"],
            scenario["sku_df"],
            scenario["budget"],
            scenario["storage_capacity"],
        )
        for scenario in operational_scenarios
    ],
    ignore_index=True,
)
operational_request_records = [
    scenario_to_request_record(
        scenario["scenario_id"],
        scenario["sku_df"],
        scenario["budget"],
        scenario["storage_capacity"],
    )
    for scenario in operational_scenarios
]

spark.createDataFrame(operational_sku_table).write.mode("overwrite").saveAsTable(sku_table_name)
spark.createDataFrame(operational_request_records).write.mode("overwrite").saveAsTable(request_table_name)
spark.sql(
    f"COMMENT ON TABLE {sku_table_name} IS 'Normalized SKU-level replenishment inputs for OR-ops batch optimization.'"
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


def solve_with_ortools(
    scenario_id: str,
    sku_df: pd.DataFrame,
    budget: int,
    storage_capacity: int,
    *,
    config_name: str,
    time_limit_s: float,
    num_workers: int,
    relative_gap: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    model = cp_model.CpModel()
    on_hand = sku_df["on_hand_cases"].astype(int).to_numpy()
    demand = sku_df["forecast_cases"].astype(int).to_numpy()
    max_order = sku_df["max_order_cases"].astype(int).to_numpy()
    unit_cost = sku_df["unit_cost"].astype(int).to_numpy()
    unit_margin = sku_df["unit_margin"].astype(int).to_numpy()
    holding_cost = sku_df["holding_cost"].astype(int).to_numpy()
    stockout_penalty = sku_df["stockout_penalty"].astype(int).to_numpy()
    storage_units = sku_df["storage_units_per_case"].astype(int).to_numpy()

    order_vars = [model.NewIntVar(0, int(max_order[index]), f"order_{index}") for index in range(len(sku_df))]
    sell_vars = [model.NewIntVar(0, int(demand[index]), f"sell_{index}") for index in range(len(sku_df))]
    ending_inventory_vars = [
        model.NewIntVar(0, int(on_hand[index] + max_order[index]), f"ending_inventory_{index}")
        for index in range(len(sku_df))
    ]
    shortage_vars = [model.NewIntVar(0, int(demand[index]), f"shortage_{index}") for index in range(len(sku_df))]

    for index in range(len(sku_df)):
        # Demand is either fulfilled or left short.
        model.Add(sell_vars[index] + shortage_vars[index] == int(demand[index]))
        # Available inventory splits into sold units and ending inventory.
        model.Add(int(on_hand[index]) + order_vars[index] == sell_vars[index] + ending_inventory_vars[index])

    model.Add(sum(int(unit_cost[index]) * order_vars[index] for index in range(len(sku_df))) <= int(budget))
    model.Add(
        sum(int(storage_units[index]) * order_vars[index] for index in range(len(sku_df))) <= int(storage_capacity)
    )
    # Reward fulfilled demand while penalizing leftover inventory and stockouts.
    model.Maximize(
        sum(
            int(unit_margin[index]) * sell_vars[index]
            - int(holding_cost[index]) * ending_inventory_vars[index]
            - int(stockout_penalty[index]) * shortage_vars[index]
            for index in range(len(sku_df))
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.relative_gap_limit = float(relative_gap)

    started = perf_counter()
    status_code = solver.Solve(model)
    solve_time_ms = (perf_counter() - started) * 1000

    status_lookup = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status = status_lookup.get(status_code, f"STATUS_{status_code}")
    is_feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    order_cases = np.array([solver.Value(var) for var in order_vars], dtype=int)
    sell_cases = np.array([solver.Value(var) for var in sell_vars], dtype=int)
    ending_inventory = np.array([solver.Value(var) for var in ending_inventory_vars], dtype=int)
    shortage_cases = np.array([solver.Value(var) for var in shortage_vars], dtype=int)

    return summarize_solution(
        scenario_id=scenario_id,
        sku_df=sku_df,
        budget=budget,
        storage_capacity=storage_capacity,
        library="ortools_cp_sat",
        config_name=config_name,
        status=status,
        solve_time_ms=solve_time_ms,
        order_cases=order_cases,
        sell_cases=sell_cases,
        ending_inventory=ending_inventory,
        shortage_cases=shortage_cases,
        is_feasible=is_feasible,
        is_optimal=status_code == cp_model.OPTIMAL,
    )


def solve_with_scipy(
    scenario_id: str,
    sku_df: pd.DataFrame,
    budget: int,
    storage_capacity: int,
    *,
    config_name: str,
    time_limit_s: float,
    mip_rel_gap: float,
    presolve: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    on_hand = sku_df["on_hand_cases"].astype(int).to_numpy()
    demand = sku_df["forecast_cases"].astype(int).to_numpy()
    max_order = sku_df["max_order_cases"].astype(int).to_numpy()
    unit_cost = sku_df["unit_cost"].astype(int).to_numpy()
    unit_margin = sku_df["unit_margin"].astype(int).to_numpy()
    holding_cost = sku_df["holding_cost"].astype(int).to_numpy()
    stockout_penalty = sku_df["stockout_penalty"].astype(int).to_numpy()
    storage_units = sku_df["storage_units_per_case"].astype(int).to_numpy()

    item_count = len(sku_df)
    order_offset = 0
    sell_offset = item_count
    ending_inventory_offset = item_count * 2
    shortage_offset = item_count * 3
    total_vars = item_count * 4

    coefficients = np.concatenate(
        [
            np.zeros(item_count, dtype=float),
            -unit_margin.astype(float),
            holding_cost.astype(float),
            stockout_penalty.astype(float),
        ]
    )
    lower_bounds = np.zeros(total_vars, dtype=float)
    upper_bounds = np.concatenate(
        [
            max_order.astype(float),
            demand.astype(float),
            (on_hand + max_order).astype(float),
            demand.astype(float),
        ]
    )

    rows: list[np.ndarray] = []
    row_lbs: list[float] = []
    row_ubs: list[float] = []

    for index in range(item_count):
        demand_row = np.zeros(total_vars, dtype=float)
        demand_row[sell_offset + index] = 1.0
        demand_row[shortage_offset + index] = 1.0
        rows.append(demand_row)
        row_lbs.append(float(demand[index]))
        row_ubs.append(float(demand[index]))

        inventory_row = np.zeros(total_vars, dtype=float)
        inventory_row[order_offset + index] = 1.0
        inventory_row[sell_offset + index] = -1.0
        inventory_row[ending_inventory_offset + index] = -1.0
        rows.append(inventory_row)
        row_lbs.append(float(-on_hand[index]))
        row_ubs.append(float(-on_hand[index]))

    budget_row = np.zeros(total_vars, dtype=float)
    budget_row[order_offset:sell_offset] = unit_cost.astype(float)
    rows.append(budget_row)
    row_lbs.append(-np.inf)
    row_ubs.append(float(budget))

    capacity_row = np.zeros(total_vars, dtype=float)
    capacity_row[order_offset:sell_offset] = storage_units.astype(float)
    rows.append(capacity_row)
    row_lbs.append(-np.inf)
    row_ubs.append(float(storage_capacity))

    started = perf_counter()
    result = milp(
        c=coefficients,
        integrality=np.ones(total_vars, dtype=int),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(row_lbs), np.asarray(row_ubs)),
        options={
            "time_limit": float(time_limit_s),
            "mip_rel_gap": float(mip_rel_gap),
            "presolve": bool(presolve),
        },
    )
    solve_time_ms = (perf_counter() - started) * 1000

    rounded = np.rint(result.x).astype(int) if result.x is not None else np.zeros(total_vars, dtype=int)
    order_cases = rounded[order_offset:sell_offset]
    sell_cases = rounded[sell_offset:ending_inventory_offset]
    ending_inventory = rounded[ending_inventory_offset:shortage_offset]
    shortage_cases = rounded[shortage_offset:]
    is_feasible = bool(result.x is not None and int(result.status) in (0, 1))

    status_lookup = {
        0: "OPTIMAL",
        1: "LIMIT_REACHED",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "OTHER",
    }
    return summarize_solution(
        scenario_id=scenario_id,
        sku_df=sku_df,
        budget=budget,
        storage_capacity=storage_capacity,
        library="scipy_milp",
        config_name=config_name,
        status=status_lookup.get(result.status, f"STATUS_{result.status}"),
        solve_time_ms=solve_time_ms,
        order_cases=order_cases,
        sell_cases=sell_cases,
        ending_inventory=ending_inventory,
        shortage_cases=shortage_cases,
        is_feasible=is_feasible,
        is_optimal=result.status == 0,
    )


def benchmark_config(config: dict[str, object], scenarios: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[str, object]]:
    scenario_rows = []
    for scenario in scenarios:
        if config["library"] == "ortools_cp_sat":
            record, _ = solve_with_ortools(
                scenario["scenario_id"],
                scenario["sku_df"],
                scenario["budget"],
                scenario["storage_capacity"],
                config_name=config["name"],
                **config["params"],
            )
        else:
            record, _ = solve_with_scipy(
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


def solve_network_with_scipy(
    network: dict[str, object],
    *,
    config_name: str,
    time_limit_s: float,
    presolve: bool = True,
) -> tuple[dict[str, object], pd.DataFrame]:
    payload = network["payload"]
    products = network["products"].reset_index(drop=True)
    sources = network["sources"].reset_index(drop=True)
    dcs = network["dcs"].reset_index(drop=True)
    stores = network["stores"].reset_index(drop=True)
    demand = network["demand"].reset_index(drop=True)
    source_dc_lanes = network["source_dc_lanes"].reset_index(drop=True)
    dc_store_lanes = network["dc_store_lanes"].reset_index(drop=True)

    product_ids = products["product_id"].tolist()
    source_ids = sources["source_id"].tolist()
    dc_ids = dcs["dc_id"].tolist()
    store_ids = stores["store_id"].tolist()
    product_count = len(product_ids)
    source_dc_lane_count = len(source_dc_lanes)
    dc_store_lane_count = len(dc_store_lanes)
    store_count = len(store_ids)

    sd_offset = 0
    ds_offset = product_count * source_dc_lane_count
    shortage_offset = ds_offset + product_count * dc_store_lane_count
    variable_count = shortage_offset + product_count * store_count

    def sd_var(product_index: int, lane_index: int) -> int:
        return sd_offset + product_index * source_dc_lane_count + lane_index

    def ds_var(product_index: int, lane_index: int) -> int:
        return ds_offset + product_index * dc_store_lane_count + lane_index

    def shortage_var(product_index: int, store_index: int) -> int:
        return shortage_offset + product_index * store_count + store_index

    product_factor = products["handling_factor"].to_numpy(dtype=float)
    shortage_penalty = products["shortage_penalty"].to_numpy(dtype=float)
    source_capacity = dict(zip(sources["source_id"], sources["capacity_cases"].astype(float)))
    dc_capacity = dict(zip(dcs["dc_id"], dcs["throughput_capacity_cases"].astype(float)))
    demand_lookup = {
        (row.product_id, row.store_id): float(row.demand_cases)
        for row in demand.itertuples(index=False)
    }

    objective = np.zeros(variable_count, dtype=float)
    sd_cost = source_dc_lanes["cost_per_case"].to_numpy(dtype=float)
    ds_cost = dc_store_lanes["cost_per_case"].to_numpy(dtype=float)
    for product_index in range(product_count):
        objective[sd_var(product_index, 0) : sd_var(product_index, source_dc_lane_count - 1) + 1] = (
            sd_cost * product_factor[product_index]
        )
        objective[ds_var(product_index, 0) : ds_var(product_index, dc_store_lane_count - 1) + 1] = (
            ds_cost * product_factor[product_index]
        )
        for store_index in range(store_count):
            objective[shortage_var(product_index, store_index)] = shortage_penalty[product_index]

    source_dc_by_source = {
        source_id: source_dc_lanes.index[source_dc_lanes["source_id"] == source_id].to_numpy(dtype=int)
        for source_id in source_ids
    }
    source_dc_by_dc = {
        dc_id: source_dc_lanes.index[source_dc_lanes["dc_id"] == dc_id].to_numpy(dtype=int)
        for dc_id in dc_ids
    }
    dc_store_by_dc = {
        dc_id: dc_store_lanes.index[dc_store_lanes["dc_id"] == dc_id].to_numpy(dtype=int)
        for dc_id in dc_ids
    }
    dc_store_by_store = {
        store_id: dc_store_lanes.index[dc_store_lanes["store_id"] == store_id].to_numpy(dtype=int)
        for store_id in store_ids
    }

    ub_rows = []
    ub_cols = []
    ub_data = []
    ub_rhs = []
    row_index = 0
    for source_id in source_ids:
        for product_index in range(product_count):
            for lane_index in source_dc_by_source[source_id]:
                ub_rows.append(row_index)
                ub_cols.append(sd_var(product_index, int(lane_index)))
                ub_data.append(1.0)
        ub_rhs.append(source_capacity[source_id])
        row_index += 1

    for dc_id in dc_ids:
        for product_index in range(product_count):
            for lane_index in dc_store_by_dc[dc_id]:
                ub_rows.append(row_index)
                ub_cols.append(ds_var(product_index, int(lane_index)))
                ub_data.append(1.0)
        ub_rhs.append(dc_capacity[dc_id])
        row_index += 1

    for product_index, _product_id in enumerate(product_ids):
        for dc_id in dc_ids:
            for lane_index in dc_store_by_dc[dc_id]:
                ub_rows.append(row_index)
                ub_cols.append(ds_var(product_index, int(lane_index)))
                ub_data.append(1.0)
            for lane_index in source_dc_by_dc[dc_id]:
                ub_rows.append(row_index)
                ub_cols.append(sd_var(product_index, int(lane_index)))
                ub_data.append(-1.0)
            ub_rhs.append(0.0)
            row_index += 1

    a_ub = sparse.csr_matrix((ub_data, (ub_rows, ub_cols)), shape=(row_index, variable_count))

    eq_rows = []
    eq_cols = []
    eq_data = []
    eq_rhs = []
    row_index = 0
    for product_index, product_id in enumerate(product_ids):
        for store_index, store_id in enumerate(store_ids):
            for lane_index in dc_store_by_store[store_id]:
                eq_rows.append(row_index)
                eq_cols.append(ds_var(product_index, int(lane_index)))
                eq_data.append(1.0)
            eq_rows.append(row_index)
            eq_cols.append(shortage_var(product_index, store_index))
            eq_data.append(1.0)
            eq_rhs.append(demand_lookup[(product_id, store_id)])
            row_index += 1

    a_eq = sparse.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(row_index, variable_count))

    started = perf_counter()
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.asarray(ub_rhs, dtype=float),
        A_eq=a_eq,
        b_eq=np.asarray(eq_rhs, dtype=float),
        bounds=(0, None),
        method="highs",
        options={"time_limit": float(time_limit_s), "presolve": bool(presolve)},
    )
    solve_time_ms = (perf_counter() - started) * 1000
    solution = np.asarray(result.x if result.x is not None else np.zeros(variable_count), dtype=float)

    flows = []
    total_source_dc_cost = 0.0
    total_dc_store_cost = 0.0
    total_shortage_cost = 0.0
    fulfilled_cases = 0.0
    shortage_cases = 0.0
    for product_index, product_id in enumerate(product_ids):
        for lane_index, lane in source_dc_lanes.iterrows():
            quantity = float(solution[sd_var(product_index, int(lane_index))])
            if quantity > 1e-6:
                unit_cost = float(lane["cost_per_case"]) * product_factor[product_index]
                cost = quantity * unit_cost
                total_source_dc_cost += cost
                flows.append(
                    {
                        "flow_type": "source_to_dc",
                        "product_id": product_id,
                        "source_id": lane["source_id"],
                        "dc_id": lane["dc_id"],
                        "store_id": None,
                        "quantity_cases": quantity,
                        "unit_cost": unit_cost,
                        "cost": cost,
                    }
                )
        for lane_index, lane in dc_store_lanes.iterrows():
            quantity = float(solution[ds_var(product_index, int(lane_index))])
            if quantity > 1e-6:
                unit_cost = float(lane["cost_per_case"]) * product_factor[product_index]
                cost = quantity * unit_cost
                total_dc_store_cost += cost
                fulfilled_cases += quantity
                flows.append(
                    {
                        "flow_type": "dc_to_store",
                        "product_id": product_id,
                        "source_id": None,
                        "dc_id": lane["dc_id"],
                        "store_id": lane["store_id"],
                        "quantity_cases": quantity,
                        "unit_cost": unit_cost,
                        "cost": cost,
                    }
                )
        for store_index, store_id in enumerate(store_ids):
            quantity = float(solution[shortage_var(product_index, store_index)])
            if quantity > 1e-6:
                unit_cost = shortage_penalty[product_index]
                cost = quantity * unit_cost
                shortage_cases += quantity
                total_shortage_cost += cost
                flows.append(
                    {
                        "flow_type": "shortage",
                        "product_id": product_id,
                        "source_id": None,
                        "dc_id": None,
                        "store_id": store_id,
                        "quantity_cases": quantity,
                        "unit_cost": unit_cost,
                        "cost": cost,
                    }
                )

    total_demand = float(demand["demand_cases"].sum())
    total_cost = total_source_dc_cost + total_dc_store_cost + total_shortage_cost
    demand_balance_residual_cases = fulfilled_cases + shortage_cases - total_demand
    summary = {
        "scenario_id": payload["scenario_id"],
        "library": "scipy_linprog_highs",
        "config_name": config_name,
        "problem_type": "distribution_network_lp",
        "status": str(result.message).split(".")[0],
        "solve_time_ms": float(solve_time_ms),
        "objective_value": float(-total_cost),
        "total_cost": float(total_cost),
        "source_dc_cost": float(total_source_dc_cost),
        "dc_store_cost": float(total_dc_store_cost),
        "shortage_cost": float(total_shortage_cost),
        "is_feasible": int(bool(result.success)),
        "is_optimal": int(result.status == 0),
        "fill_rate": float(min(1.0, max(0.0, fulfilled_cases / total_demand)) if total_demand else 0.0),
        "shortage_rate": float(shortage_cases / total_demand if total_demand else 0.0),
        "demand_balance_residual_cases": float(demand_balance_residual_cases),
        "total_demand": total_demand,
        "fulfilled_cases": float(fulfilled_cases),
        "shortage_cases": float(shortage_cases),
        "product_count": int(product_count),
        "source_count": int(len(source_ids)),
        "dc_count": int(len(dc_ids)),
        "store_count": int(len(store_ids)),
        "source_dc_lane_count": int(source_dc_lane_count),
        "dc_store_lane_count": int(dc_store_lane_count),
        "variable_count": int(variable_count),
        "constraint_count": int(a_ub.shape[0] + a_eq.shape[0]),
        "matrix_nonzero_count": int(a_ub.nnz + a_eq.nnz),
        "flow_count": int(len(flows)),
        "time_limit_s": float(time_limit_s),
    }
    return summary, pd.DataFrame(flows)


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
        for key, value in config["params"].items():
            mlflow.log_param(f"solver__{key}", value)
        mlflow.log_metrics(mlflow_metric_dict(summary))
        mlflow.log_table(scenario_frame, artifact_file=f"benchmark/{config['name']}_scenario_results.json")


def select_champion(summary_frame: pd.DataFrame) -> dict[str, object]:
    champion_row = summary_frame.iloc[0].to_dict()
    champion_row["selection_rule"] = "max feasible_ratio, max avg_fill_rate, max avg_objective, min avg_solve_time_ms"
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
    model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model=str(model_script_path),
        registered_model_name=registered_model_name,
        signature=model_signature,
        input_example=[example_request],
        pip_requirements=build_model_requirements(champion_config["library"]),
    )
    model_version = resolve_logged_model_version(registered_model_name, run_id)
    return model_info, model_version


def build_model_requirements(library: str) -> list[str]:
    requirements = [
        f"mlflow[databricks]=={metadata.version('mlflow')}",
        f"pydantic=={metadata.version('pydantic')}",
        f"typing_extensions=={metadata.version('typing_extensions')}",
    ]
    if library == "ortools_cp_sat":
        requirements.append(f"ortools=={metadata.version('ortools')}")
    else:
        requirements.extend([f"numpy=={metadata.version('numpy')}", f"scipy=={metadata.version('scipy')}"])
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


def create_or_update_endpoint(endpoint_name: str, registered_model_name: str, model_version: str) -> dict[str, str]:
    workspace = WorkspaceClient()
    served_model_name = f"{registered_model_name.split('.')[-1]}-{model_version}"
    served_entities = [
        ServedEntityInput(
            entity_name=registered_model_name,
            entity_version=str(model_version),
            name=served_model_name,
            workload_size="Small",
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
        )
        action = "created"
    else:
        workspace.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=served_entities,
            traffic_config=traffic_config,
        )
        action = "updated"

    return {
        "action": action,
        "endpoint_name": endpoint_name,
        "registered_model_name": registered_model_name,
        "model_version": str(model_version),
        "served_model_name": served_model_name,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Solve the small example first
# MAGIC
# MAGIC Start with one explainable scenario before running the full benchmark. This makes it easier to sanity-check the objective, the capacity constraints, and the shape of the recommended order plan.

# COMMAND ----------

example_record, example_solution = solve_with_ortools(
    example_scenario_id,
    example_sku_df,
    example_budget,
    example_storage_capacity,
    config_name="ortools_baseline_demo",
    time_limit_s=4.0,
    num_workers=1,
    relative_gap=0.0,
)

display(example_sku_table.sort_values(["category", "sku_id"]).reset_index(drop=True))
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
# MAGIC ## 4. Define the benchmark sweep
# MAGIC
# MAGIC The benchmark below sweeps several scenarios and solver settings against the same replenishment contract. The goal is not just to find the absolute highest objective: pick the configuration that stays feasible across scenarios while keeping solve times practical for the planning cadence.

# COMMAND ----------

sku_counts = np.asarray(small_sku_counts, dtype=int)
benchmark_scenarios = []
for index, sku_count in enumerate(sku_counts, start=1):
    scenario_id = f"week_{index:02d}_{sku_count}skus"
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run the MLflow experiment and register the champion
# MAGIC
# MAGIC One MLflow experiment is the shared scoreboard for the same business problem. Different users, agents, and solver libraries can each log a run against the same scenario inputs, and the experiment becomes the single place to compare them, pick a winner, and promote it under governance.
# MAGIC
# MAGIC The MLflow pattern for OR work is:
# MAGIC
# MAGIC 1. run the solver sweep against the same scenario inputs
# MAGIC 2. log solver parameters, objective metrics, and scenario-level artifacts so every attempt is comparable later
# MAGIC 3. select a champion with an explicit rule that the team agrees on
# MAGIC 4. register that champion as a governed Unity Catalog model version
# MAGIC 5. validate the registered model locally before exposing it through any access pattern

# COMMAND ----------

run_name = f"inventory_benchmark_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"
mlflow.set_experiment(experiment_name)

with TemporaryDirectory() as temp_dir:
    temp_root = Path(temp_dir)
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(
            {
                "problem_type": "inventory_replenishment",
                "scenario_count": scenario_count,
                "seed": seed,
                "catalog": catalog,
                "schema": schema,
                "model_name": model_name,
                "registered_model_name": registered_model_name,
                "deploy_endpoint": int(deploy_endpoint),
            }
        )
        mlflow.log_dict(example_request, "artifacts/input_example.json")
        mlflow.log_table(example_solution, artifact_file="artifacts/example_solution.json")

        summary_frame, scenario_frames = run_benchmark(solver_configs, benchmark_scenarios)
        for config in solver_configs:
            summary = summary_frame[summary_frame["config_name"] == config["name"]].iloc[0].to_dict()
            log_solver_run(config, scenario_frames[config["name"]], summary)

        champion_row = select_champion(summary_frame)
        champion_config = next(config for config in solver_configs if config["name"] == champion_row["config_name"])

        mlflow.log_table(summary_frame, artifact_file="benchmark/solver_comparison.json")
        mlflow.log_dict(champion_row, "benchmark/champion.json")
        mlflow.log_metrics(
            {
                f"champion_{key}": value
                for key, value in mlflow_metric_dict(champion_row).items()
                if key not in {"scenario_count"}
            }
        )
        mlflow.set_tag("champion_library", champion_row["library"])
        mlflow.set_tag("champion_config_name", champion_row["config_name"])

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
    "champion": champion_row,
    "champion_config": champion_config,
    "registered_model_name": registered_model_name,
    "registered_model_version": model_version,
    "model_uri": model_info.model_uri,
    "validation_prediction": validation_preview,
}
print(json.dumps(experiment_result, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5b. Optional large-scale CPU vs GPU benchmark
# MAGIC
# MAGIC The small benchmark above is the production-model selection path; the OR contract there is intentionally compact so the entire flow stays readable.
# MAGIC
# MAGIC OR teams also want to know how the same MLflow tracking surface answers a different question: at what scale does it pay off to switch runtimes? This optional section logs a separate stress experiment on a larger, sparse distribution-network LP (product flows from sources to DCs to stores, with source capacity, DC throughput, demand at every store, and shortage penalties). The companion GPU notebook generates the same network shape from the same seed, so the two runs land in the same MLflow experiment and can be compared apples-to-apples.

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
    large_run_name = f"large_distribution_cpu_{large_benchmark_id}"
    with mlflow.start_run(run_name=large_run_name) as large_active_run:
        mlflow.log_params(
            {
                "benchmark_id": large_benchmark_id,
                "benchmark_mode": "large_scale_cpu_gpu",
                "accelerator": "serverless_cpu",
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
                "accelerator": "serverless_cpu",
                "benchmark_scope": "large_cpu_gpu_comparison",
            }
        )

        large_record, large_flows = solve_network_with_scipy(
            network,
            config_name="scipy_network_sparse_lp",
            time_limit_s=large_time_limit_s,
            presolve=True,
        )
        large_record = {
            **large_record,
            "benchmark_id": large_benchmark_id,
            "benchmark_mode": "large_scale_cpu_gpu",
            "mlflow_run_id": large_active_run.info.run_id,
            "scenario_seed": large_scenario_seed,
            "accelerator": "serverless_cpu",
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
            accelerator="serverless_cpu",
        )

        with mlflow.start_run(run_name="scipy_network_sparse_lp", nested=True):
            mlflow.log_param("library", large_record["library"])
            mlflow.log_param("accelerator", "serverless_cpu")
            mlflow.log_param("benchmark_id", large_benchmark_id)
            mlflow.log_param("benchmark_mode", "large_scale_cpu_gpu")
            mlflow.log_param("time_limit_s", large_time_limit_s)
            mlflow.log_metrics(finite_mlflow_metric_dict(large_record))
            mlflow.log_table(pd.DataFrame([large_record]), artifact_file="large_benchmark/cpu_summary.json")

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
# MAGIC ## 6. Choose the right OR-ops access pattern
# MAGIC
# MAGIC The same registered MLflow model supports several access paths. Pick the one that fits the use case before wiring up downstream code; the next sections each implement one of these paths against the same Champion model artifact.
# MAGIC
# MAGIC | Pattern | Input shape | Best for |
# MAGIC | --- | --- | --- |
# MAGIC | Spark `applyInPandas` | SKU-long table grouped by `scenario_id` | Scheduled batch optimization, Delta outputs, replayable jobs |
# MAGIC | Model Serving | One JSON request per scenario | Apps, what-if workflows, external integrations |
# MAGIC | SQL `ai_query` | One array-backed request row per scenario | Analyst workflows and endpoint-backed SQL batch calls |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Run Spark batch optimization with the Champion model
# MAGIC
# MAGIC Most OR workloads run on a planning cadence: read the latest inputs, solve every scenario, and write recommendation tables that downstream systems and auditors can consume.
# MAGIC
# MAGIC The Champion model registered above is the single artifact every access pattern reuses. In Spark, each group is one scenario from the normalized SKU table; the grouped pandas function builds the request payload, loads the `Champion` model once per Python worker, and emits one recommendation row per SKU. Nothing about the optimization logic is duplicated here — it is the same governed model that serving and `ai_query` will hit.

# COMMAND ----------

batch_recommendation_schema = StructType(
    [
        StructField("scenario_id", StringType(), False),
        StructField("sku_id", StringType(), False),
        StructField("sku_sequence", LongType(), False),
        StructField("category", StringType(), True),
        StructField("status", StringType(), False),
        StructField("is_feasible", BooleanType(), False),
        StructField("is_optimal", BooleanType(), False),
        StructField("library", StringType(), False),
        StructField("config_name", StringType(), False),
        StructField("objective_value", DoubleType(), False),
        StructField("fill_rate", DoubleType(), False),
        StructField("total_order_spend", DoubleType(), False),
        StructField("total_storage_used", DoubleType(), False),
        StructField("budget_slack", DoubleType(), False),
        StructField("storage_slack", DoubleType(), False),
        StructField("forecast_cases", LongType(), False),
        StructField("on_hand_cases", LongType(), False),
        StructField("order_cases", LongType(), False),
        StructField("sell_cases", LongType(), False),
        StructField("ending_inventory_cases", LongType(), False),
        StructField("shortage_cases", LongType(), False),
    ]
)


def solve_scenario_with_champion(pdf: pd.DataFrame) -> pd.DataFrame:
    import mlflow
    import pandas as pd

    ordered = pdf.sort_values("sku_sequence").reset_index(drop=True)
    request = {
        "scenario_id": str(ordered["scenario_id"].iloc[0]),
        "sku_ids": ordered["sku_id"].astype(str).tolist(),
        "on_hand": ordered["on_hand_cases"].astype(int).tolist(),
        "forecast": ordered["forecast_cases"].astype(int).tolist(),
        "unit_cost": ordered["unit_cost"].astype(int).tolist(),
        "unit_margin": ordered["unit_margin"].astype(int).tolist(),
        "holding_cost": ordered["holding_cost"].astype(int).tolist(),
        "stockout_penalty": ordered["stockout_penalty"].astype(int).tolist(),
        "storage_units": ordered["storage_units_per_case"].astype(int).tolist(),
        "max_order": ordered["max_order_cases"].astype(int).tolist(),
        "budget": int(ordered["budget"].iloc[0]),
        "storage_capacity": int(ordered["storage_capacity"].iloc[0]),
    }

    def rows_from_response(response: dict) -> pd.DataFrame:
        rows = ordered[
            [
                "scenario_id",
                "sku_id",
                "sku_sequence",
                "category",
                "forecast_cases",
                "on_hand_cases",
            ]
        ].copy()
        rows["order_cases"] = 0
        rows["shortage_cases"] = 0
        rows["sell_cases"] = rows["forecast_cases"].astype(int)
        rows["ending_inventory_cases"] = (rows["on_hand_cases"].astype(int) - rows["forecast_cases"].astype(int)).clip(
            lower=0
        )

        recommendations = pd.DataFrame(response.get("recommendations", []))
        if not recommendations.empty:
            rows = rows.drop(columns=["order_cases", "sell_cases", "ending_inventory_cases", "shortage_cases"]).merge(
                recommendations,
                on="sku_id",
                how="left",
            )
            rows["order_cases"] = rows["order_cases"].fillna(0)
            rows["shortage_cases"] = rows["shortage_cases"].fillna(0)
            rows["sell_cases"] = rows["sell_cases"].fillna(rows["forecast_cases"])
            default_ending_inventory = (rows["on_hand_cases"].astype(int) - rows["forecast_cases"].astype(int)).clip(
                lower=0
            )
            rows["ending_inventory_cases"] = rows["ending_inventory_cases"].fillna(default_ending_inventory)
        return rows

    def rows_from_solution(response: dict, solution: pd.DataFrame) -> pd.DataFrame:
        rows = solution[
            [
                "scenario_id",
                "sku_id",
                "sku_sequence",
                "category",
                "forecast_cases",
                "on_hand_cases",
                "order_cases",
                "sell_cases",
                "ending_inventory_cases",
                "shortage_cases",
            ]
        ].copy()
        rows["library"] = str(response["library"])
        rows["config_name"] = str(response["config_name"])
        return rows

    try:
        if not getattr(solve_scenario_with_champion, "_model_load_failed", False):
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            if not hasattr(solve_scenario_with_champion, "_model"):
                solve_scenario_with_champion._model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}@Champion")
            prediction = solve_scenario_with_champion._model.predict([request])
            response = prediction[0] if isinstance(prediction, list) else prediction.to_dict(orient="records")[0]
            rows = rows_from_response(response)
        else:
            raise RuntimeError("Skipping model registry load after a previous worker auth failure.")
    except Exception:
        solve_scenario_with_champion._model_load_failed = True
        if champion_config["library"] == "ortools_cp_sat":
            response, solution = solve_with_ortools(
                request["scenario_id"],
                ordered,
                request["budget"],
                request["storage_capacity"],
                config_name=champion_config["name"],
                **champion_config["params"],
            )
        elif champion_config["library"] == "scipy_milp":
            response, solution = solve_with_scipy(
                request["scenario_id"],
                ordered,
                request["budget"],
                request["storage_capacity"],
                config_name=champion_config["name"],
                **champion_config["params"],
            )
        else:
            raise ValueError(f"Unsupported batch solver library: {champion_config['library']}")
        rows = rows_from_solution(response, solution)

    rows = rows[
        [
            "scenario_id",
            "sku_id",
            "sku_sequence",
            "category",
            "forecast_cases",
            "on_hand_cases",
            "order_cases",
            "sell_cases",
            "ending_inventory_cases",
            "shortage_cases",
        ]
    ].copy()

    for column in ["forecast_cases", "on_hand_cases", "order_cases", "sell_cases", "ending_inventory_cases", "shortage_cases"]:
        rows[column] = rows[column].astype(int)

    rows["status"] = str(response["status"])
    rows["is_feasible"] = bool(response["is_feasible"])
    rows["is_optimal"] = bool(response["is_optimal"])
    rows["library"] = str(response["library"])
    rows["config_name"] = str(response["config_name"])
    rows["objective_value"] = float(response["objective_value"])
    rows["fill_rate"] = float(response["fill_rate"])
    rows["total_order_spend"] = float(response["total_order_spend"])
    rows["total_storage_used"] = float(response["total_storage_used"])
    rows["budget_slack"] = float(response["budget_slack"])
    rows["storage_slack"] = float(response["storage_slack"])
    return rows[
        [
            "scenario_id",
            "sku_id",
            "sku_sequence",
            "category",
            "status",
            "is_feasible",
            "is_optimal",
            "library",
            "config_name",
            "objective_value",
            "fill_rate",
            "total_order_spend",
            "total_storage_used",
            "budget_slack",
            "storage_slack",
            "forecast_cases",
            "on_hand_cases",
            "order_cases",
            "sell_cases",
            "ending_inventory_cases",
            "shortage_cases",
        ]
    ]


batch_recommendations = spark.table(sku_table_name).groupBy("scenario_id").applyInPandas(
    solve_scenario_with_champion,
    schema=batch_recommendation_schema,
)
batch_recommendations.write.mode("overwrite").saveAsTable(recommendation_table_name)

scenario_result_columns = [
    "scenario_id",
    "status",
    "is_feasible",
    "is_optimal",
    "library",
    "config_name",
    "objective_value",
    "fill_rate",
    "total_order_spend",
    "total_storage_used",
    "budget_slack",
    "storage_slack",
]
scenario_results = spark.table(recommendation_table_name).select(*scenario_result_columns).dropDuplicates(["scenario_id"])
scenario_results.write.mode("overwrite").saveAsTable(scenario_result_table_name)
spark.sql(
    f"COMMENT ON TABLE {recommendation_table_name} IS 'SKU-level replenishment recommendations produced by the registered Champion optimizer.'"
)
spark.sql(
    f"COMMENT ON TABLE {scenario_result_table_name} IS 'Scenario-level optimization outcomes produced by the registered Champion optimizer.'"
)
batch_result = {
    "scenario_result_table_name": scenario_result_table_name,
    "recommendation_table_name": recommendation_table_name,
    "scenario_count": int(scenario_results.count()),
    "recommendation_row_count": int(spark.table(recommendation_table_name).count()),
}

display(spark.table(scenario_result_table_name).orderBy("scenario_id"))
display(spark.table(recommendation_table_name).orderBy("scenario_id", "sku_sequence"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Deploy the champion to Model Serving
# MAGIC
# MAGIC Serving is the interactive OR-ops access path. The same Champion model that the Spark batch path just used is exposed behind an HTTP endpoint, so apps and analysts can solve a single scenario on demand without rerunning the benchmark or reloading the artifact.
# MAGIC
# MAGIC Creating or updating a serverless Model Serving endpoint can take up to 20 minutes; rerun this cell on its own when you want to refresh the endpoint without rerunning the benchmark.

# COMMAND ----------

deployment_result = {
    "skipped": True,
    "reason": "Set deploy_endpoint=true to create or update the serving endpoint.",
}

if deploy_endpoint:
    deployment_result = create_or_update_endpoint(
        endpoint_name,
        experiment_result["registered_model_name"],
        experiment_result["registered_model_version"],
    )
    with mlflow.start_run(run_id=experiment_result["run_id"]):
        mlflow.log_dict(deployment_result, "deployment/endpoint_result.json")

notebook_result = {
    **experiment_result,
    "batch_result": batch_result,
    "deployment_result": deployment_result,
    "large_benchmark_result": large_benchmark_summary,
}
print(json.dumps(notebook_result, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Review the comparison and the promoted model
# MAGIC
# MAGIC `summary_frame` is the decision table for promotion. `notebook_result` captures the winning config, the registered model version, the batch output tables, and the optional endpoint status.

# COMMAND ----------

display(summary_frame)
display(pd.DataFrame([experiment_result["champion"]]))
print(json.dumps(notebook_result, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Query the deployed endpoint from Python
# MAGIC
# MAGIC The serving request uses `dataframe_records` with one scenario request row. That keeps the REST path aligned with SQL `ai_query`, which invokes custom model endpoints with the same tabular record shape.

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
# MAGIC ## 11. Query the same endpoint from SQL with `ai_query`
# MAGIC
# MAGIC The same governed endpoint can also be invoked from Databricks SQL. This is useful for analyst-facing workflows where inputs already exist as one request row per scenario. For large scheduled optimization jobs, prefer the Spark batch path above because it avoids one endpoint call per scenario.

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
# MAGIC ## 12. What this OR-ops workflow produced
# MAGIC
# MAGIC The core lifecycle is complete at this point:
# MAGIC
# MAGIC - one MLflow experiment with nested runs for each solver configuration that any team member or downstream agent can extend with another attempt
# MAGIC - one ranked comparison table and champion selection rule
# MAGIC - one Unity Catalog registered model version with the `Champion` alias
# MAGIC - normalized input and request snapshot tables in Unity Catalog
# MAGIC - scenario-level and SKU-level optimization output tables
# MAGIC - one optional serverless serving endpoint
# MAGIC - Spark batch, Python/REST, and SQL `ai_query` invocation patterns that reuse the same governed model artifact
# MAGIC - one optional, separate large-scale benchmark experiment for comparing CPU vs GPU runtimes on a sparse distribution-network LP

# COMMAND ----------

final_summary = {
    "experiment_name": experiment_result["experiment_name"],
    "run_id": experiment_result["run_id"],
    "registered_model_name": experiment_result["registered_model_name"],
    "registered_model_version": experiment_result["registered_model_version"],
    "champion_config_name": experiment_result["champion"]["config_name"],
    "champion_library": experiment_result["champion"]["library"],
    "endpoint_name": deployment_result.get("endpoint_name", endpoint_name),
    "endpoint_action": deployment_result.get("action", "skipped"),
    "large_benchmark_result": large_benchmark_summary,
    "recommendation_table_name": recommendation_table_name,
    "request_table_name": request_table_name,
    "scenario_result_table_name": scenario_result_table_name,
    "sku_table_name": sku_table_name,
}
print(json.dumps(final_summary, indent=2, sort_keys=True))
