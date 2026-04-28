import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mlflow.models import set_model
from mlflow.pyfunc import PythonModel
from pydantic import BaseModel, Field

MODEL_LIBRARY = "__MODEL_LIBRARY__"
MODEL_CONFIG_NAME = "__MODEL_CONFIG_NAME__"
MODEL_PARAMS = json.loads(r'''__MODEL_PARAMS_JSON__''')


class InventoryRequest(BaseModel):
    scenario_id: str = Field(description="Identifier for the replenishment scenario.")
    sku_ids: list[str] = Field(description="Ordered list of SKU identifiers.")
    on_hand: list[int] = Field(description="Cases currently on hand for each SKU.")
    forecast: list[int] = Field(description="Forecast demand in cases for each SKU.")
    unit_cost: list[int] = Field(description="Procurement cost per case for each SKU.")
    unit_margin: list[int] = Field(description="Contribution margin per fulfilled case.")
    holding_cost: list[int] = Field(description="Penalty per leftover case.")
    stockout_penalty: list[int] = Field(description="Penalty per unmet case of demand.")
    storage_units: list[int] = Field(description="Storage units consumed per ordered case.")
    max_order: list[int] = Field(description="Maximum order quantity per SKU.")
    budget: int = Field(description="Weekly procurement budget in abstract currency units.")
    storage_capacity: int = Field(description="Weekly storage capacity in abstract storage units.")


def _validate_lengths(request: InventoryRequest) -> int:
    lengths = {
        "sku_ids": len(request.sku_ids),
        "on_hand": len(request.on_hand),
        "forecast": len(request.forecast),
        "unit_cost": len(request.unit_cost),
        "unit_margin": len(request.unit_margin),
        "holding_cost": len(request.holding_cost),
        "stockout_penalty": len(request.stockout_penalty),
        "storage_units": len(request.storage_units),
        "max_order": len(request.max_order),
    }
    distinct_lengths = set(lengths.values())
    if len(distinct_lengths) != 1:
        raise ValueError(f"All scenario arrays must have the same length. Got: {lengths}")
    return next(iter(distinct_lengths))


def _coerce_request_record(record: Any) -> InventoryRequest:
    if isinstance(record, InventoryRequest):
        return record
    if hasattr(record, "asDict"):
        record = record.asDict(recursive=True)
    elif hasattr(record, "to_dict") and not isinstance(record, dict):
        record = record.to_dict()
    return InventoryRequest.model_validate(record)


def _coerce_requests(model_input: Any) -> list[InventoryRequest]:
    if hasattr(model_input, "to_dict") and hasattr(model_input, "columns"):
        records = model_input.to_dict(orient="records")
    elif isinstance(model_input, dict):
        raw_records = model_input.get("inputs", model_input.get("dataframe_records"))
        if raw_records is None:
            records = [model_input]
        elif isinstance(raw_records, list):
            records = raw_records
        else:
            records = [raw_records]
    elif isinstance(model_input, InventoryRequest):
        records = [model_input]
    elif isinstance(model_input, Iterable) and not isinstance(model_input, (str, bytes)):
        records = list(model_input)
    else:
        records = [model_input]
    return [_coerce_request_record(record) for record in records]


def _build_response(
    *,
    request: InventoryRequest,
    status: str,
    is_feasible: bool,
    is_optimal: bool,
    order_cases: list[int],
    sell_cases: list[int],
    ending_inventory: list[int],
    shortage_cases: list[int],
) -> dict[str, Any]:
    item_count = _validate_lengths(request)
    recommendations = [
        {
            "sku_id": request.sku_ids[index],
            "order_cases": int(order_cases[index]),
            "sell_cases": int(sell_cases[index]),
            "ending_inventory_cases": int(ending_inventory[index]),
            "shortage_cases": int(shortage_cases[index]),
        }
        for index in range(item_count)
        if int(order_cases[index]) > 0 or int(shortage_cases[index]) > 0
    ]
    gross_margin_reward = float(
        sum(int(sell_cases[index]) * int(request.unit_margin[index]) for index in range(item_count))
    )
    holding_cost_penalty = float(
        sum(int(ending_inventory[index]) * int(request.holding_cost[index]) for index in range(item_count))
    )
    stockout_penalty_cost = float(
        sum(int(shortage_cases[index]) * int(request.stockout_penalty[index]) for index in range(item_count))
    )
    objective_value = gross_margin_reward - holding_cost_penalty - stockout_penalty_cost
    total_demand = max(sum(int(value) for value in request.forecast), 1)
    total_shortage_cases = sum(int(value) for value in shortage_cases)
    total_order_spend = float(
        sum(int(order_cases[index]) * int(request.unit_cost[index]) for index in range(item_count))
    )
    total_storage_used = float(
        sum(int(order_cases[index]) * int(request.storage_units[index]) for index in range(item_count))
    )
    return {
        "scenario_id": request.scenario_id,
        "library": MODEL_LIBRARY,
        "config_name": MODEL_CONFIG_NAME,
        "status": status,
        "is_feasible": bool(is_feasible),
        "is_optimal": bool(is_optimal),
        "objective_value": objective_value,
        "gross_margin_reward": gross_margin_reward,
        "holding_cost_penalty": holding_cost_penalty,
        "stockout_penalty_cost": stockout_penalty_cost,
        "fill_rate": float(sum(int(value) for value in sell_cases) / total_demand),
        "shortage_rate": float(total_shortage_cases / total_demand),
        "total_order_spend": total_order_spend,
        "total_storage_used": total_storage_used,
        "budget_slack": float(request.budget - total_order_spend),
        "storage_slack": float(request.storage_capacity - total_storage_used),
        "ordered_sku_count": int(sum(1 for value in order_cases if int(value) > 0)),
        "total_shortage_cases": int(total_shortage_cases),
        "recommendations": recommendations,
    }


def _solve_with_ortools(request: InventoryRequest) -> dict[str, Any]:
    from ortools.sat.python import cp_model

    item_count = _validate_lengths(request)
    model = cp_model.CpModel()
    order_vars = [
        model.NewIntVar(0, int(request.max_order[index]), f"order_{index}") for index in range(item_count)
    ]
    sell_vars = [
        model.NewIntVar(0, int(request.forecast[index]), f"sell_{index}") for index in range(item_count)
    ]
    ending_inventory_vars = [
        model.NewIntVar(0, int(request.on_hand[index] + request.max_order[index]), f"ending_inventory_{index}")
        for index in range(item_count)
    ]
    shortage_vars = [
        model.NewIntVar(0, int(request.forecast[index]), f"shortage_{index}") for index in range(item_count)
    ]

    for index in range(item_count):
        model.Add(sell_vars[index] + shortage_vars[index] == int(request.forecast[index]))
        model.Add(int(request.on_hand[index]) + order_vars[index] == sell_vars[index] + ending_inventory_vars[index])

    model.Add(sum(int(request.unit_cost[index]) * order_vars[index] for index in range(item_count)) <= int(request.budget))
    model.Add(
        sum(int(request.storage_units[index]) * order_vars[index] for index in range(item_count))
        <= int(request.storage_capacity)
    )
    model.Maximize(
        sum(
            int(request.unit_margin[index]) * sell_vars[index]
            - int(request.holding_cost[index]) * ending_inventory_vars[index]
            - int(request.stockout_penalty[index]) * shortage_vars[index]
            for index in range(item_count)
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(MODEL_PARAMS.get("time_limit_s", 4.0))
    solver.parameters.num_search_workers = int(MODEL_PARAMS.get("num_workers", 8))
    solver.parameters.relative_gap_limit = float(MODEL_PARAMS.get("relative_gap", 0.0))
    status_code = solver.Solve(model)

    status_lookup = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    return _build_response(
        request=request,
        status=status_lookup.get(status_code, f"STATUS_{status_code}"),
        is_feasible=status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE),
        is_optimal=status_code == cp_model.OPTIMAL,
        order_cases=[solver.Value(var) for var in order_vars],
        sell_cases=[solver.Value(var) for var in sell_vars],
        ending_inventory=[solver.Value(var) for var in ending_inventory_vars],
        shortage_cases=[solver.Value(var) for var in shortage_vars],
    )


def _solve_with_scipy(request: InventoryRequest) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    item_count = _validate_lengths(request)
    on_hand = np.asarray(request.on_hand, dtype=int)
    demand = np.asarray(request.forecast, dtype=int)
    unit_cost = np.asarray(request.unit_cost, dtype=int)
    unit_margin = np.asarray(request.unit_margin, dtype=int)
    holding_cost = np.asarray(request.holding_cost, dtype=int)
    stockout_penalty = np.asarray(request.stockout_penalty, dtype=int)
    storage_units = np.asarray(request.storage_units, dtype=int)
    max_order = np.asarray(request.max_order, dtype=int)

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
    row_ubs.append(float(request.budget))

    capacity_row = np.zeros(total_vars, dtype=float)
    capacity_row[order_offset:sell_offset] = storage_units.astype(float)
    rows.append(capacity_row)
    row_lbs.append(-np.inf)
    row_ubs.append(float(request.storage_capacity))

    result = milp(
        c=coefficients,
        integrality=np.ones(total_vars, dtype=int),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(row_lbs), np.asarray(row_ubs)),
        options={
            "time_limit": float(MODEL_PARAMS.get("time_limit_s", 4.0)),
            "mip_rel_gap": float(MODEL_PARAMS.get("mip_rel_gap", 0.0)),
            "presolve": bool(MODEL_PARAMS.get("presolve", True)),
        },
    )

    rounded = np.rint(result.x).astype(int) if result.x is not None else np.zeros(total_vars, dtype=int)
    order_cases = rounded[order_offset:sell_offset].tolist()
    sell_cases = rounded[sell_offset:ending_inventory_offset].tolist()
    ending_inventory = rounded[ending_inventory_offset:shortage_offset].tolist()
    shortage_cases = rounded[shortage_offset:].tolist()

    status_lookup = {
        0: "OPTIMAL",
        1: "LIMIT_REACHED",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "OTHER",
    }
    return _build_response(
        request=request,
        status=status_lookup.get(int(result.status), f"STATUS_{result.status}"),
        is_feasible=bool(result.x is not None and int(result.status) in (0, 1)),
        is_optimal=int(result.status) == 0,
        order_cases=order_cases,
        sell_cases=sell_cases,
        ending_inventory=ending_inventory,
        shortage_cases=shortage_cases,
    )


def _solve_with_cuopt(request: InventoryRequest) -> dict[str, Any]:
    import importlib.util
    import os
    import subprocess
    import sys

    _validate_lengths(request)
    payload = request.model_dump()
    payload["time_limit_s"] = float(MODEL_PARAMS.get("time_limit_s", 4.0))

    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONPATH"] = os.pathsep.join(
        path for path in [*sys.path, subprocess_env.get("PYTHONPATH", "")] if path
    )
    helper_spec = importlib.util.find_spec("cuopt_inventory_subprocess")
    subprocess_command = [sys.executable, "-m", "cuopt_inventory_subprocess"]
    if helper_spec is not None and helper_spec.origin:
        helper_path = Path(helper_spec.origin).resolve()
        if helper_path.exists():
            subprocess_command = [sys.executable, str(helper_path)]

    completed = subprocess.run(
        subprocess_command,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=subprocess_env,
        timeout=max(120, int(payload["time_limit_s"]) + 120),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "cuOpt failed in an isolated Python subprocess. "
            "This usually means the native CUDA/cuOpt libraries are incompatible with the current GPU runtime. "
            f"Return code: {completed.returncode}\n"
            f"stderr tail:\n{completed.stderr[-4000:]}\n"
            f"stdout tail:\n{completed.stdout[-4000:]}"
        )

    result_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("CUOPT_RESULT_JSON:")),
        None,
    )
    if result_line is None:
        raise RuntimeError(f"cuOpt subprocess finished without a parseable result. stdout:\n{completed.stdout[-4000:]}")
    result = json.loads(result_line.removeprefix("CUOPT_RESULT_JSON:"))

    return _build_response(
        request=request,
        status=result["status"],
        is_feasible=bool(result["is_feasible"]),
        is_optimal=bool(result["is_optimal"]),
        order_cases=result["order_cases"],
        sell_cases=result["sell_cases"],
        ending_inventory=result["ending_inventory"],
        shortage_cases=result["shortage_cases"],
    )


class InventoryOptimizerModel(PythonModel):
    def predict(self, model_input, params=None) -> list[dict[str, Any]]:
        requests = _coerce_requests(model_input)
        if MODEL_LIBRARY == "ortools_cp_sat":
            return [_solve_with_ortools(request) for request in requests]
        if MODEL_LIBRARY == "scipy_milp":
            return [_solve_with_scipy(request) for request in requests]
        if MODEL_LIBRARY == "cuopt_milp":
            return [_solve_with_cuopt(request) for request in requests]
        raise ValueError(f"Unsupported library: {MODEL_LIBRARY}")


set_model(InventoryOptimizerModel())
