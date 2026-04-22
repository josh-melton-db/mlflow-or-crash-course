# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow + Inventory Optimization Crash Course
# MAGIC
# MAGIC This notebook is a compact, blog-friendly walkthrough for experimenting with operations research on Databricks:
# MAGIC
# MAGIC 1. Generate synthetic inventory replenishment scenarios for a distribution center.
# MAGIC 2. Compare `OR-Tools CP-SAT` and `SciPy milp` across multiple solver settings.
# MAGIC 3. Track every run in MLflow.
# MAGIC 4. Select the champion configuration using business-friendly metrics like fill rate and solve time.
# MAGIC 5. Package the winning solver as an MLflow Model From Code.
# MAGIC 6. Optionally deploy the registered model to Databricks Model Serving on serverless compute.

# COMMAND ----------

# MAGIC %pip install -q -U mlflow==3.11.1 ortools==9.15.6755 scipy==1.15.3 pandas==2.3.3 numpy==2.2.6 pyarrow==23.0.1 databricks-sdk==0.103.0

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "or_blog_josh_melton")
dbutils.widgets.text("experiment_name", "")
dbutils.widgets.text("registered_model_name", "")
dbutils.widgets.text("endpoint_name", "inventory-optimizer-endpoint")
dbutils.widgets.text("scenario_count", "6")
dbutils.widgets.text("seed", "7")
dbutils.widgets.dropdown("deploy_endpoint", "true", ["true", "false"])

# COMMAND ----------

import importlib.metadata as metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlflow
import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, Route, ServedEntityInput, TrafficConfig
from mlflow import MlflowClient
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, milp

# COMMAND ----------

catalog = dbutils.widgets.get("catalog").strip() or "main"
schema = dbutils.widgets.get("schema").strip() or "or_blog_josh_melton"
endpoint_name = dbutils.widgets.get("endpoint_name").strip() or "inventory-optimizer-endpoint"
scenario_count = max(3, int(dbutils.widgets.get("scenario_count") or "6"))
seed = int(dbutils.widgets.get("seed") or "7")
deploy_endpoint = dbutils.widgets.get("deploy_endpoint").strip().lower() == "true"

current_user = spark.sql("SELECT current_user()").first()[0]
experiment_name = dbutils.widgets.get("experiment_name").strip() or f"/Users/{current_user}/inventory-optimization-crash-course"
registered_model_name = (
    dbutils.widgets.get("registered_model_name").strip() or f"{catalog}.{schema}.inventory_optimizer"
)

spark.sql(
    f"""
    CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`
    COMMENT 'Inventory optimization crash course assets'
    """
)

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_name)

run_context = {
    "catalog": catalog,
    "schema": schema,
    "experiment_name": experiment_name,
    "registered_model_name": registered_model_name,
    "endpoint_name": endpoint_name,
    "scenario_count": scenario_count,
    "seed": seed,
    "deploy_endpoint": deploy_endpoint,
}
print(json.dumps(run_context, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem setup
# MAGIC
# MAGIC Each SKU belongs to a distribution center replenishment plan. The optimizer decides how many cases to order this week.
# MAGIC
# MAGIC Decision variables:
# MAGIC
# MAGIC - `order_cases_i`: integer cases to reorder
# MAGIC - `sell_cases_i`: cases that satisfy demand
# MAGIC - `ending_inventory_i`: leftover cases after demand is met
# MAGIC - `shortage_cases_i`: unmet demand
# MAGIC
# MAGIC Business constraints:
# MAGIC
# MAGIC - procurement budget
# MAGIC - storage capacity in abstract storage units
# MAGIC - maximum order quantity per SKU
# MAGIC
# MAGIC Objective:
# MAGIC
# MAGIC Maximize contribution margin from fulfilled demand while penalizing leftover inventory and stockouts.

# COMMAND ----------

CATEGORY_POOL = [
    "ambient_snacks",
    "beverages",
    "canned_goods",
    "cereal",
    "cleaning",
    "condiments",
    "frozen_meals",
    "personal_care",
    "produce",
    "dairy",
]


def generate_inventory_scenario(scenario_id: str, sku_count: int, scenario_seed: int) -> tuple[pd.DataFrame, int, int]:
    rng = np.random.default_rng(scenario_seed)

    unit_cost = rng.integers(8, 42, size=sku_count)
    unit_margin = rng.integers(5, 18, size=sku_count)
    on_hand_cases = rng.integers(4, 36, size=sku_count)
    forecast_cases = rng.integers(14, 90, size=sku_count)
    max_order_cases = rng.integers(12, 65, size=sku_count)
    holding_cost = rng.integers(1, 4, size=sku_count)
    stockout_penalty = rng.integers(7, 21, size=sku_count)
    storage_units_per_case = rng.integers(1, 5, size=sku_count)

    sku_df = pd.DataFrame(
        {
            "sku_id": [f"{scenario_id.upper()}_SKU_{index + 1:03d}" for index in range(sku_count)],
            "category": rng.choice(CATEGORY_POOL, size=sku_count, replace=True),
            "on_hand_cases": on_hand_cases,
            "forecast_cases": forecast_cases,
            "unit_cost": unit_cost,
            "unit_margin": unit_margin,
            "holding_cost": holding_cost,
            "stockout_penalty": stockout_penalty,
            "storage_units_per_case": storage_units_per_case,
            "max_order_cases": max_order_cases,
        }
    )

    budget = int((sku_df["unit_cost"] * sku_df["max_order_cases"]).sum() * rng.uniform(0.36, 0.52))
    storage_capacity = int(
        (sku_df["storage_units_per_case"] * sku_df["max_order_cases"]).sum() * rng.uniform(0.38, 0.54)
    )
    return sku_df, budget, storage_capacity


def scenario_to_request_row(
    scenario_id: str,
    sku_df: pd.DataFrame,
    budget: int,
    storage_capacity: int,
) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "sku_ids_json": json.dumps(sku_df["sku_id"].tolist()),
        "on_hand_json": json.dumps(sku_df["on_hand_cases"].astype(int).tolist()),
        "forecast_json": json.dumps(sku_df["forecast_cases"].astype(int).tolist()),
        "unit_cost_json": json.dumps(sku_df["unit_cost"].astype(int).tolist()),
        "unit_margin_json": json.dumps(sku_df["unit_margin"].astype(int).tolist()),
        "holding_cost_json": json.dumps(sku_df["holding_cost"].astype(int).tolist()),
        "stockout_penalty_json": json.dumps(sku_df["stockout_penalty"].astype(int).tolist()),
        "storage_units_json": json.dumps(sku_df["storage_units_per_case"].astype(int).tolist()),
        "max_order_json": json.dumps(sku_df["max_order_cases"].astype(int).tolist()),
        "budget": int(budget),
        "storage_capacity": int(storage_capacity),
    }


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
    is_optimal: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    result_df = sku_df.copy()
    result_df["order_cases"] = order_cases.astype(int)
    result_df["sell_cases"] = sell_cases.astype(int)
    result_df["ending_inventory_cases"] = ending_inventory.astype(int)
    result_df["shortage_cases"] = shortage_cases.astype(int)
    result_df["order_spend"] = result_df["order_cases"] * result_df["unit_cost"]
    result_df["storage_used"] = result_df["order_cases"] * result_df["storage_units_per_case"]
    result_df["objective_component"] = (
        result_df["sell_cases"] * result_df["unit_margin"]
        - result_df["ending_inventory_cases"] * result_df["holding_cost"]
        - result_df["shortage_cases"] * result_df["stockout_penalty"]
    )

    total_demand = int(result_df["forecast_cases"].sum())
    total_sold = int(result_df["sell_cases"].sum())
    total_order_spend = float(result_df["order_spend"].sum())
    total_storage_used = float(result_df["storage_used"].sum())

    record = {
        "scenario_id": scenario_id,
        "sku_count": int(len(result_df)),
        "library": library,
        "config_name": config_name,
        "status": status,
        "solve_time_ms": float(solve_time_ms),
        "objective_value": float(result_df["objective_component"].sum()),
        "is_feasible": int(1),
        "is_optimal": int(is_optimal),
        "fill_rate": float(total_sold / total_demand),
        "budget_utilization": float(total_order_spend / budget),
        "storage_utilization": float(total_storage_used / storage_capacity),
        "ordered_sku_count": int((result_df["order_cases"] > 0).sum()),
        "total_order_cases": int(result_df["order_cases"].sum()),
        "total_order_spend": total_order_spend,
        "total_storage_used": total_storage_used,
        "total_shortage_cases": int(result_df["shortage_cases"].sum()),
    }
    return record, result_df


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
    leftover_vars = [
        model.NewIntVar(0, int(on_hand[index] + max_order[index]), f"leftover_{index}") for index in range(len(sku_df))
    ]
    shortage_vars = [model.NewIntVar(0, int(demand[index]), f"shortage_{index}") for index in range(len(sku_df))]

    for index in range(len(sku_df)):
        model.Add(sell_vars[index] + shortage_vars[index] == int(demand[index]))
        model.Add(int(on_hand[index]) + order_vars[index] == sell_vars[index] + leftover_vars[index])

    model.Add(sum(int(unit_cost[index]) * order_vars[index] for index in range(len(sku_df))) <= int(budget))
    model.Add(
        sum(int(storage_units[index]) * order_vars[index] for index in range(len(sku_df))) <= int(storage_capacity)
    )
    model.Maximize(
        sum(
            int(unit_margin[index]) * sell_vars[index]
            - int(holding_cost[index]) * leftover_vars[index]
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
    order_cases = np.array([solver.Value(var) for var in order_vars], dtype=int)
    sell_cases = np.array([solver.Value(var) for var in sell_vars], dtype=int)
    leftover_cases = np.array([solver.Value(var) for var in leftover_vars], dtype=int)
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
        ending_inventory=leftover_cases,
        shortage_cases=shortage_cases,
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
    leftover_offset = item_count * 2
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
        inventory_row[leftover_offset + index] = -1.0
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
    sell_cases = rounded[sell_offset:leftover_offset]
    leftover_cases = rounded[leftover_offset:shortage_offset]
    shortage_cases = rounded[shortage_offset:]

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
        ending_inventory=leftover_cases,
        shortage_cases=shortage_cases,
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
    summary = {
        "config_name": config["name"],
        "library": config["library"],
        "avg_objective": float(frame["objective_value"].mean()),
        "avg_solve_time_ms": float(frame["solve_time_ms"].mean()),
        "feasible_ratio": float(frame["is_feasible"].mean()),
        "optimal_ratio": float(frame["is_optimal"].mean()),
        "avg_fill_rate": float(frame["fill_rate"].mean()),
        "avg_budget_utilization": float(frame["budget_utilization"].mean()),
        "avg_storage_utilization": float(frame["storage_utilization"].mean()),
        "avg_ordered_sku_count": float(frame["ordered_sku_count"].mean()),
        "scenario_count": int(len(frame)),
    }
    for key, value in config["params"].items():
        summary[f"param__{key}"] = value
    return frame, summary


def build_model_requirements(library: str) -> list[str]:
    base_packages = ["mlflow", "pandas"]
    if library == "ortools_cp_sat":
        base_packages.append("ortools")
    else:
        base_packages.extend(["numpy", "scipy"])
    return [f"{package}=={metadata.version(package)}" for package in base_packages]


def render_model_script(champion_config: dict[str, object], output_path: Path) -> Path:
    library = champion_config["library"]
    params_repr = repr(champion_config["params"])
    model_name = champion_config["name"]

    if library == "ortools_cp_sat":
        script = f"""
import json
from typing import Any

import pandas as pd
from mlflow.models import set_model
from mlflow.pyfunc import PythonModel
from ortools.sat.python import cp_model

MODEL_LIBRARY = {library!r}
MODEL_CONFIG_NAME = {model_name!r}
MODEL_PARAMS = {params_repr}


def _coerce_sequence(value: Any, field_name: str) -> list[int]:
    if value is None:
        raise ValueError(f"Missing required field: {{field_name}}")
    if isinstance(value, str):
        return [int(item) if not isinstance(item, str) else item for item in json.loads(value)]
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _solve_row(row: dict[str, Any]) -> dict[str, Any]:
    sku_ids = [str(item) for item in _coerce_sequence(row.get("sku_ids_json"), "sku_ids_json")]
    on_hand = [int(item) for item in _coerce_sequence(row.get("on_hand_json"), "on_hand_json")]
    demand = [int(item) for item in _coerce_sequence(row.get("forecast_json"), "forecast_json")]
    unit_cost = [int(item) for item in _coerce_sequence(row.get("unit_cost_json"), "unit_cost_json")]
    unit_margin = [int(item) for item in _coerce_sequence(row.get("unit_margin_json"), "unit_margin_json")]
    holding_cost = [int(item) for item in _coerce_sequence(row.get("holding_cost_json"), "holding_cost_json")]
    stockout_penalty = [int(item) for item in _coerce_sequence(row.get("stockout_penalty_json"), "stockout_penalty_json")]
    storage_units = [int(item) for item in _coerce_sequence(row.get("storage_units_json"), "storage_units_json")]
    max_order = [int(item) for item in _coerce_sequence(row.get("max_order_json"), "max_order_json")]
    budget = int(row.get("budget"))
    storage_capacity = int(row.get("storage_capacity"))

    model = cp_model.CpModel()
    order_vars = [model.NewIntVar(0, max_order[index], f"order_{{index}}") for index in range(len(sku_ids))]
    sell_vars = [model.NewIntVar(0, demand[index], f"sell_{{index}}") for index in range(len(sku_ids))]
    leftover_vars = [
        model.NewIntVar(0, on_hand[index] + max_order[index], f"leftover_{{index}}")
        for index in range(len(sku_ids))
    ]
    shortage_vars = [model.NewIntVar(0, demand[index], f"shortage_{{index}}") for index in range(len(sku_ids))]

    for index in range(len(sku_ids)):
        model.Add(sell_vars[index] + shortage_vars[index] == demand[index])
        model.Add(on_hand[index] + order_vars[index] == sell_vars[index] + leftover_vars[index])

    model.Add(sum(unit_cost[index] * order_vars[index] for index in range(len(sku_ids))) <= budget)
    model.Add(sum(storage_units[index] * order_vars[index] for index in range(len(sku_ids))) <= storage_capacity)
    model.Maximize(
        sum(
            unit_margin[index] * sell_vars[index]
            - holding_cost[index] * leftover_vars[index]
            - stockout_penalty[index] * shortage_vars[index]
            for index in range(len(sku_ids))
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(MODEL_PARAMS.get("time_limit_s", 4.0))
    solver.parameters.num_search_workers = int(MODEL_PARAMS.get("num_workers", 8))
    solver.parameters.relative_gap_limit = float(MODEL_PARAMS.get("relative_gap", 0.0))

    status_code = solver.Solve(model)
    order_cases = [solver.Value(var) for var in order_vars]
    sell_cases = [solver.Value(var) for var in sell_vars]
    leftover_cases = [solver.Value(var) for var in leftover_vars]
    shortage_cases = [solver.Value(var) for var in shortage_vars]

    status_lookup = {{
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }}

    return {{
        "scenario_id": str(row.get("scenario_id") or "serving_request"),
        "library": MODEL_LIBRARY,
        "config_name": MODEL_CONFIG_NAME,
        "status": status_lookup.get(status_code, f"STATUS_{{status_code}}"),
        "is_feasible": bool(status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)),
        "is_optimal": bool(status_code == cp_model.OPTIMAL),
        "objective_value": float(
            sum(
                sell_cases[index] * unit_margin[index]
                - leftover_cases[index] * holding_cost[index]
                - shortage_cases[index] * stockout_penalty[index]
                for index in range(len(sku_ids))
            )
        ),
        "fill_rate": float(sum(sell_cases) / max(sum(demand), 1)),
        "total_order_spend": float(sum(order_cases[index] * unit_cost[index] for index in range(len(sku_ids)))),
        "total_storage_used": float(
            sum(order_cases[index] * storage_units[index] for index in range(len(sku_ids)))
        ),
        "ordered_sku_count": int(sum(1 for value in order_cases if value > 0)),
        "recommended_orders_json": json.dumps(
            [
                {{
                    "sku_id": sku_ids[index],
                    "order_cases": int(order_cases[index]),
                    "sell_cases": int(sell_cases[index]),
                    "ending_inventory_cases": int(leftover_cases[index]),
                    "shortage_cases": int(shortage_cases[index]),
                }}
                for index in range(len(sku_ids))
                if order_cases[index] > 0 or shortage_cases[index] > 0
            ]
        ),
    }}


class InventoryOptimizerModel(PythonModel):
    def predict(self, context, model_input, params=None):
        frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        return pd.DataFrame([_solve_row(record) for record in frame.to_dict(orient="records")])


set_model(InventoryOptimizerModel())
"""
    else:
        script = f"""
import json
from typing import Any

import numpy as np
import pandas as pd
from mlflow.models import set_model
from mlflow.pyfunc import PythonModel
from scipy.optimize import Bounds, LinearConstraint, milp

MODEL_LIBRARY = {library!r}
MODEL_CONFIG_NAME = {model_name!r}
MODEL_PARAMS = {params_repr}


def _coerce_sequence(value: Any, field_name: str) -> list[int]:
    if value is None:
        raise ValueError(f"Missing required field: {{field_name}}")
    if isinstance(value, str):
        return [int(item) if not isinstance(item, str) else item for item in json.loads(value)]
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _solve_row(row: dict[str, Any]) -> dict[str, Any]:
    sku_ids = [str(item) for item in _coerce_sequence(row.get("sku_ids_json"), "sku_ids_json")]
    on_hand = np.asarray([int(item) for item in _coerce_sequence(row.get("on_hand_json"), "on_hand_json")], dtype=int)
    demand = np.asarray([int(item) for item in _coerce_sequence(row.get("forecast_json"), "forecast_json")], dtype=int)
    unit_cost = np.asarray([int(item) for item in _coerce_sequence(row.get("unit_cost_json"), "unit_cost_json")], dtype=int)
    unit_margin = np.asarray([int(item) for item in _coerce_sequence(row.get("unit_margin_json"), "unit_margin_json")], dtype=int)
    holding_cost = np.asarray([int(item) for item in _coerce_sequence(row.get("holding_cost_json"), "holding_cost_json")], dtype=int)
    stockout_penalty = np.asarray([int(item) for item in _coerce_sequence(row.get("stockout_penalty_json"), "stockout_penalty_json")], dtype=int)
    storage_units = np.asarray([int(item) for item in _coerce_sequence(row.get("storage_units_json"), "storage_units_json")], dtype=int)
    max_order = np.asarray([int(item) for item in _coerce_sequence(row.get("max_order_json"), "max_order_json")], dtype=int)
    budget = int(row.get("budget"))
    storage_capacity = int(row.get("storage_capacity"))

    item_count = len(sku_ids)
    order_offset = 0
    sell_offset = item_count
    leftover_offset = item_count * 2
    shortage_offset = item_count * 3
    total_vars = item_count * 4

    coefficients = np.concatenate([
        np.zeros(item_count, dtype=float),
        -unit_margin.astype(float),
        holding_cost.astype(float),
        stockout_penalty.astype(float),
    ])
    lower_bounds = np.zeros(total_vars, dtype=float)
    upper_bounds = np.concatenate([
        max_order.astype(float),
        demand.astype(float),
        (on_hand + max_order).astype(float),
        demand.astype(float),
    ])

    rows = []
    row_lbs = []
    row_ubs = []
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
        inventory_row[leftover_offset + index] = -1.0
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

    result = milp(
        c=coefficients,
        integrality=np.ones(total_vars, dtype=int),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(row_lbs), np.asarray(row_ubs)),
        options={{
            "time_limit": float(MODEL_PARAMS.get("time_limit_s", 4.0)),
            "mip_rel_gap": float(MODEL_PARAMS.get("mip_rel_gap", 0.0)),
            "presolve": bool(MODEL_PARAMS.get("presolve", True)),
        }},
    )

    rounded = np.rint(result.x).astype(int) if result.x is not None else np.zeros(total_vars, dtype=int)
    order_cases = rounded[order_offset:sell_offset]
    sell_cases = rounded[sell_offset:leftover_offset]
    leftover_cases = rounded[leftover_offset:shortage_offset]
    shortage_cases = rounded[shortage_offset:]

    status_lookup = {{
        0: "OPTIMAL",
        1: "LIMIT_REACHED",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "OTHER",
    }}

    return {{
        "scenario_id": str(row.get("scenario_id") or "serving_request"),
        "library": MODEL_LIBRARY,
        "config_name": MODEL_CONFIG_NAME,
        "status": status_lookup.get(result.status, f"STATUS_{{result.status}}"),
        "is_feasible": True,
        "is_optimal": bool(result.status == 0),
        "objective_value": float(
            np.sum(sell_cases * unit_margin - leftover_cases * holding_cost - shortage_cases * stockout_penalty)
        ),
        "fill_rate": float(np.sum(sell_cases) / max(np.sum(demand), 1)),
        "total_order_spend": float(np.sum(order_cases * unit_cost)),
        "total_storage_used": float(np.sum(order_cases * storage_units)),
        "ordered_sku_count": int(np.sum(order_cases > 0)),
        "recommended_orders_json": json.dumps(
            [
                {{
                    "sku_id": sku_ids[index],
                    "order_cases": int(order_cases[index]),
                    "sell_cases": int(sell_cases[index]),
                    "ending_inventory_cases": int(leftover_cases[index]),
                    "shortage_cases": int(shortage_cases[index]),
                }}
                for index in range(item_count)
                if order_cases[index] > 0 or shortage_cases[index] > 0
            ]
        ),
    }}


class InventoryOptimizerModel(PythonModel):
    def predict(self, context, model_input, params=None):
        frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        return pd.DataFrame([_solve_row(record) for record in frame.to_dict(orient="records")])


set_model(InventoryOptimizerModel())
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script.strip() + "\n", encoding="utf-8")
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
        workspace.serving_endpoints.get(endpoint_name)
        workspace.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=served_entities,
            traffic_config=traffic_config,
        )
        action = "updated"
    except Exception:
        workspace.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=served_entities,
                traffic_config=traffic_config,
            ),
        )
        action = "created"

    return {
        "action": action,
        "endpoint_name": endpoint_name,
        "registered_model_name": registered_model_name,
        "model_version": str(model_version),
        "served_model_name": served_model_name,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example scenario
# MAGIC
# MAGIC This smaller scenario is useful in the blog post because it keeps the optimization tables readable.

# COMMAND ----------

example_scenario_id = "week_demo"
example_sku_df, example_budget, example_storage_capacity = generate_inventory_scenario(
    example_scenario_id,
    sku_count=12,
    scenario_seed=seed,
)
example_record, example_solution = solve_with_ortools(
    example_scenario_id,
    example_sku_df,
    example_budget,
    example_storage_capacity,
    config_name="ortools_demo",
    time_limit_s=4.0,
    num_workers=1,
    relative_gap=0.0,
)

display(example_sku_df.sort_values(["category", "sku_id"]).reset_index(drop=True))
display(
    example_solution.loc[
        example_solution["order_cases"] > 0,
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
        ],
    ]
    .sort_values("order_cases", ascending=False)
    .reset_index(drop=True)
)
display(pd.DataFrame([example_record]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmark solver libraries and parameter settings

# COMMAND ----------

sku_counts = np.linspace(18, 72, num=scenario_count, dtype=int)
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

summary_rows = []
run_name = f"inventory_benchmark_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"

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
                "registered_model_name": registered_model_name,
                "deploy_endpoint": int(deploy_endpoint),
            }
        )
        mlflow.log_table(
            pd.DataFrame([scenario_to_request_row(example_scenario_id, example_sku_df, example_budget, example_storage_capacity)]),
            artifact_file="artifacts/input_example.json",
        )
        mlflow.log_table(example_solution, artifact_file="artifacts/example_solution.json")

        for config in solver_configs:
            scenario_frame, summary = benchmark_config(config, benchmark_scenarios)
            summary_rows.append(summary)

            with mlflow.start_run(run_name=config["name"], nested=True):
                mlflow.log_param("library", config["library"])
                for key, value in config["params"].items():
                    mlflow.log_param(f"solver__{key}", value)
                mlflow.log_metrics(
                    {
                        "avg_objective": summary["avg_objective"],
                        "avg_solve_time_ms": summary["avg_solve_time_ms"],
                        "feasible_ratio": summary["feasible_ratio"],
                        "optimal_ratio": summary["optimal_ratio"],
                        "avg_fill_rate": summary["avg_fill_rate"],
                        "avg_budget_utilization": summary["avg_budget_utilization"],
                        "avg_storage_utilization": summary["avg_storage_utilization"],
                    }
                )
                mlflow.log_table(scenario_frame, artifact_file=f"benchmark/{config['name']}_scenario_results.json")

        summary_frame = pd.DataFrame(summary_rows).sort_values(
            by=["feasible_ratio", "avg_fill_rate", "avg_objective", "avg_solve_time_ms"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

        champion_row = summary_frame.iloc[0].to_dict()
        champion_row["selection_rule"] = "max feasible_ratio, max avg_fill_rate, max avg_objective, min avg_solve_time_ms"
        champion_config = next(config for config in solver_configs if config["name"] == champion_row["config_name"])

        mlflow.log_table(summary_frame, artifact_file="benchmark/solver_comparison.json")
        mlflow.log_dict(champion_row, "benchmark/champion.json")
        mlflow.set_tag("champion_library", champion_row["library"])
        mlflow.set_tag("champion_config_name", champion_row["config_name"])

        input_example = pd.DataFrame(
            [scenario_to_request_row(example_scenario_id, example_sku_df, example_budget, example_storage_capacity)]
        )
        model_script_path = render_model_script(champion_config, temp_root / "inventory_optimizer_model.py")
        model_info = mlflow.pyfunc.log_model(
            name="inventory_optimizer",
            python_model=str(model_script_path),
            registered_model_name=registered_model_name,
            input_example=input_example,
            pip_requirements=build_model_requirements(champion_config["library"]),
        )
        model_version = resolve_logged_model_version(registered_model_name, active_run.info.run_id)

        deployment_result = None
        if deploy_endpoint:
            deployment_result = create_or_update_endpoint(endpoint_name, registered_model_name, model_version)
            mlflow.log_dict(deployment_result, "deployment/endpoint_result.json")

        notebook_result = {
            "run_id": active_run.info.run_id,
            "experiment_name": experiment_name,
            "champion": champion_row,
            "champion_config": champion_config,
            "registered_model_name": registered_model_name,
            "registered_model_version": model_version,
            "model_uri": model_info.model_uri,
            "deployment_result": deployment_result,
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

display(summary_frame)
print(json.dumps(notebook_result, indent=2, sort_keys=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample serving payload
# MAGIC
# MAGIC Use this JSON body with `databricks serving-endpoints query` once the endpoint is ready.

# COMMAND ----------

sample_payload = {
    "dataframe_records": [
        scenario_to_request_row(example_scenario_id, example_sku_df, example_budget, example_storage_capacity)
    ]
}
print(json.dumps(sample_payload, indent=2))
