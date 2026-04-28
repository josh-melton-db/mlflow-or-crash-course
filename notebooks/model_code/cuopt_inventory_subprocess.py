import ctypes
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from time import perf_counter


def preload_nccl_libraries() -> None:
    try:
        nccl_dist = distribution("nvidia-nccl-cu12")
    except PackageNotFoundError:
        return

    for relative_path in nccl_dist.files or []:
        if str(relative_path).endswith("libnccl.so.2"):
            resolved = Path(nccl_dist.locate_file(relative_path)).resolve()
            ctypes.CDLL(str(resolved), mode=ctypes.RTLD_GLOBAL)
            return


def rows_from_payload(payload: dict) -> list[dict]:
    if "sku_rows" in payload:
        return payload["sku_rows"]

    return [
        {
            "sku_id": payload["sku_ids"][index],
            "on_hand_cases": payload["on_hand"][index],
            "forecast_cases": payload["forecast"][index],
            "unit_cost": payload["unit_cost"][index],
            "unit_margin": payload["unit_margin"][index],
            "holding_cost": payload["holding_cost"][index],
            "stockout_penalty": payload["stockout_penalty"][index],
            "storage_units_per_case": payload["storage_units"][index],
            "max_order_cases": payload["max_order"][index],
        }
        for index in range(len(payload["sku_ids"]))
    ]


def solve_inventory_payload(payload: dict) -> dict:
    preload_nccl_libraries()

    from cuopt.linear_programming.problem import INTEGER, MAXIMIZE, Problem
    from cuopt.linear_programming.solver_settings import SolverSettings

    rows = rows_from_payload(payload)
    problem = Problem(f"inventory_{payload['scenario_id']}")

    order_vars = [
        problem.addVariable(lb=0, ub=int(row["max_order_cases"]), vtype=INTEGER, name=f"order_{index}")
        for index, row in enumerate(rows)
    ]
    sell_vars = [
        problem.addVariable(lb=0, ub=int(row["forecast_cases"]), vtype=INTEGER, name=f"sell_{index}")
        for index, row in enumerate(rows)
    ]
    ending_inventory_vars = [
        problem.addVariable(
            lb=0,
            ub=int(row["on_hand_cases"] + row["max_order_cases"]),
            vtype=INTEGER,
            name=f"ending_inventory_{index}",
        )
        for index, row in enumerate(rows)
    ]
    shortage_vars = [
        problem.addVariable(lb=0, ub=int(row["forecast_cases"]), vtype=INTEGER, name=f"shortage_{index}")
        for index, row in enumerate(rows)
    ]

    for index, row in enumerate(rows):
        problem.addConstraint(
            sell_vars[index] + shortage_vars[index] == int(row["forecast_cases"]),
            name=f"demand_balance_{index}",
        )
        problem.addConstraint(
            int(row["on_hand_cases"]) + order_vars[index] == sell_vars[index] + ending_inventory_vars[index],
            name=f"inventory_balance_{index}",
        )

    problem.addConstraint(
        sum(int(row["unit_cost"]) * order_vars[index] for index, row in enumerate(rows)) <= int(payload["budget"]),
        name="budget",
    )
    problem.addConstraint(
        sum(int(row["storage_units_per_case"]) * order_vars[index] for index, row in enumerate(rows))
        <= int(payload["storage_capacity"]),
        name="storage_capacity",
    )
    problem.setObjective(
        sum(
            int(row["unit_margin"]) * sell_vars[index]
            - int(row["holding_cost"]) * ending_inventory_vars[index]
            - int(row["stockout_penalty"]) * shortage_vars[index]
            for index, row in enumerate(rows)
        ),
        sense=MAXIMIZE,
    )

    settings = SolverSettings()
    settings.set_parameter("time_limit", float(payload["time_limit_s"]))

    started = perf_counter()
    problem.solve(settings)
    solve_time_ms = (perf_counter() - started) * 1000

    status = str(problem.Status.name)
    is_feasible = status.lower() in {"optimal", "feasible"}

    def values(variables):
        if not is_feasible:
            return [0 for _ in variables]
        return [int(round(float(variable.getValue()))) for variable in variables]

    return {
        "status": status.upper(),
        "solve_time_ms": solve_time_ms,
        "is_feasible": is_feasible,
        "is_optimal": status.lower() == "optimal",
        "order_cases": values(order_vars),
        "sell_cases": values(sell_vars),
        "ending_inventory": values(ending_inventory_vars),
        "shortage_cases": values(shortage_vars) if is_feasible else [int(row["forecast_cases"]) for row in rows],
    }


def solve_network_payload(payload: dict) -> dict:
    preload_nccl_libraries()

    from cuopt.linear_programming.problem import CONTINUOUS, MINIMIZE, Problem
    from cuopt.linear_programming.solver_settings import SolverSettings

    try:
        from cuopt.linear_programming.solver.solver_parameters import CUOPT_METHOD
        from cuopt.linear_programming.solver_settings import SolverMethod
    except ImportError:
        CUOPT_METHOD = None
        SolverMethod = None

    products = payload["products"]
    sources = payload["sources"]
    dcs = payload["dcs"]
    stores = payload["stores"]
    source_dc_lanes = payload["source_dc_lanes"]
    dc_store_lanes = payload["dc_store_lanes"]
    demand = payload["demand"]
    time_limit_s = float(payload["time_limit_s"])

    source_capacity = {row["source_id"]: float(row["capacity_cases"]) for row in sources}
    dc_capacity = {row["dc_id"]: float(row["throughput_capacity_cases"]) for row in dcs}
    product_ids = [row["product_id"] for row in products]
    store_ids = [row["store_id"] for row in stores]
    product_factor = {row["product_id"]: float(row["handling_factor"]) for row in products}
    shortage_penalty = {row["product_id"]: float(row["shortage_penalty"]) for row in products}

    demand_lookup = {
        (row["product_id"], row["store_id"]): float(row["demand_cases"])
        for row in demand
    }

    problem = Problem(f"network_{payload['scenario_id']}")

    source_dc_vars = {}
    for product_id in product_ids:
        factor = product_factor[product_id]
        for lane_index, lane in enumerate(source_dc_lanes):
            source_dc_vars[(product_id, lane_index)] = problem.addVariable(
                lb=0,
                vtype=CONTINUOUS,
                name=f"ship_sd_{product_id}_{lane_index}",
            )

    dc_store_vars = {}
    for product_id in product_ids:
        for lane_index, lane in enumerate(dc_store_lanes):
            dc_store_vars[(product_id, lane_index)] = problem.addVariable(
                lb=0,
                vtype=CONTINUOUS,
                name=f"ship_ds_{product_id}_{lane_index}",
            )

    shortage_vars = {}
    for product_id in product_ids:
        for store_id in store_ids:
            shortage_vars[(product_id, store_id)] = problem.addVariable(
                lb=0,
                vtype=CONTINUOUS,
                name=f"shortage_{product_id}_{store_id}",
            )

    source_dc_by_source = {}
    source_dc_by_dc = {}
    for lane_index, lane in enumerate(source_dc_lanes):
        source_dc_by_source.setdefault(lane["source_id"], []).append(lane_index)
        source_dc_by_dc.setdefault(lane["dc_id"], []).append(lane_index)

    dc_store_by_dc = {}
    dc_store_by_store = {}
    for lane_index, lane in enumerate(dc_store_lanes):
        dc_store_by_dc.setdefault(lane["dc_id"], []).append(lane_index)
        dc_store_by_store.setdefault(lane["store_id"], []).append(lane_index)

    for source_id, lane_indices in source_dc_by_source.items():
        problem.addConstraint(
            sum(source_dc_vars[(product_id, lane_index)] for product_id in product_ids for lane_index in lane_indices)
            <= source_capacity[source_id],
            name=f"source_capacity_{source_id}",
        )

    for dc_id, lane_indices in dc_store_by_dc.items():
        problem.addConstraint(
            sum(dc_store_vars[(product_id, lane_index)] for product_id in product_ids for lane_index in lane_indices)
            <= dc_capacity[dc_id],
            name=f"dc_throughput_{dc_id}",
        )

    for product_id in product_ids:
        for dc_id in dc_capacity:
            inbound = sum(
                source_dc_vars[(product_id, lane_index)]
                for lane_index in source_dc_by_dc.get(dc_id, [])
            )
            outbound = sum(
                dc_store_vars[(product_id, lane_index)]
                for lane_index in dc_store_by_dc.get(dc_id, [])
            )
            problem.addConstraint(outbound <= inbound, name=f"flow_balance_{product_id}_{dc_id}")

    for product_id in product_ids:
        for store_id in store_ids:
            inbound = sum(
                dc_store_vars[(product_id, lane_index)]
                for lane_index in dc_store_by_store.get(store_id, [])
            )
            problem.addConstraint(
                inbound + shortage_vars[(product_id, store_id)] == demand_lookup[(product_id, store_id)],
                name=f"demand_{product_id}_{store_id}",
            )

    problem.setObjective(
        sum(
            float(lane["cost_per_case"]) * product_factor[product_id] * source_dc_vars[(product_id, lane_index)]
            for product_id in product_ids
            for lane_index, lane in enumerate(source_dc_lanes)
        )
        + sum(
            float(lane["cost_per_case"]) * product_factor[product_id] * dc_store_vars[(product_id, lane_index)]
            for product_id in product_ids
            for lane_index, lane in enumerate(dc_store_lanes)
        )
        + sum(
            shortage_penalty[product_id] * shortage_vars[(product_id, store_id)]
            for product_id in product_ids
            for store_id in store_ids
        ),
        sense=MINIMIZE,
    )

    settings = SolverSettings()
    settings.set_parameter("time_limit", time_limit_s)
    if CUOPT_METHOD is not None and SolverMethod is not None:
        try:
            settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
        except Exception:
            pass

    started = perf_counter()
    problem.solve(settings)
    solve_time_ms = (perf_counter() - started) * 1000

    status = str(problem.Status.name)
    is_feasible = status.lower() in {"optimal", "feasible"}

    flows = []
    shortage_rows = []
    total_source_dc_cost = 0.0
    total_dc_store_cost = 0.0
    total_shortage_cost = 0.0
    total_fulfilled = 0.0
    total_shortage = 0.0

    if is_feasible:
        for product_id in product_ids:
            factor = product_factor[product_id]
            for lane_index, lane in enumerate(source_dc_lanes):
                quantity = float(source_dc_vars[(product_id, lane_index)].getValue())
                if quantity > 1e-6:
                    cost = quantity * float(lane["cost_per_case"]) * factor
                    total_source_dc_cost += cost
                    flows.append(
                        {
                            "flow_type": "source_to_dc",
                            "product_id": product_id,
                            "source_id": lane["source_id"],
                            "dc_id": lane["dc_id"],
                            "store_id": None,
                            "quantity_cases": quantity,
                            "unit_cost": float(lane["cost_per_case"]) * factor,
                            "cost": cost,
                        }
                    )
            for lane_index, lane in enumerate(dc_store_lanes):
                quantity = float(dc_store_vars[(product_id, lane_index)].getValue())
                if quantity > 1e-6:
                    cost = quantity * float(lane["cost_per_case"]) * factor
                    total_dc_store_cost += cost
                    total_fulfilled += quantity
                    flows.append(
                        {
                            "flow_type": "dc_to_store",
                            "product_id": product_id,
                            "source_id": None,
                            "dc_id": lane["dc_id"],
                            "store_id": lane["store_id"],
                            "quantity_cases": quantity,
                            "unit_cost": float(lane["cost_per_case"]) * factor,
                            "cost": cost,
                        }
                    )
            for store_id in store_ids:
                quantity = float(shortage_vars[(product_id, store_id)].getValue())
                if quantity > 1e-6:
                    cost = quantity * shortage_penalty[product_id]
                    total_shortage += quantity
                    total_shortage_cost += cost
                    shortage_rows.append(
                        {
                            "flow_type": "shortage",
                            "product_id": product_id,
                            "source_id": None,
                            "dc_id": None,
                            "store_id": store_id,
                            "quantity_cases": quantity,
                            "unit_cost": shortage_penalty[product_id],
                            "cost": cost,
                        }
                    )
    else:
        total_shortage = sum(float(row["demand_cases"]) for row in demand)
        total_shortage_cost = sum(
            float(row["demand_cases"]) * shortage_penalty[row["product_id"]]
            for row in demand
        )

    total_demand = sum(float(row["demand_cases"]) for row in demand)
    total_cost = total_source_dc_cost + total_dc_store_cost + total_shortage_cost
    demand_balance_residual_cases = total_fulfilled + total_shortage - total_demand
    fill_rate = min(1.0, max(0.0, (total_demand - total_shortage) / total_demand)) if total_demand else 0.0

    return {
        "status": status.upper(),
        "solve_time_ms": solve_time_ms,
        "is_feasible": is_feasible,
        "is_optimal": status.lower() == "optimal",
        "objective_value": -total_cost,
        "total_cost": total_cost,
        "source_dc_cost": total_source_dc_cost,
        "dc_store_cost": total_dc_store_cost,
        "shortage_cost": total_shortage_cost,
        "total_demand": total_demand,
        "fulfilled_cases": total_fulfilled,
        "shortage_cases": total_shortage,
        "fill_rate": fill_rate,
        "demand_balance_residual_cases": demand_balance_residual_cases,
        "flow_count": len(flows),
        "shortage_row_count": len(shortage_rows),
        "flows": flows + shortage_rows,
    }


def solve_payload(payload: dict) -> dict:
    if payload.get("problem_type") == "distribution_network_lp":
        return solve_network_payload(payload)
    return solve_inventory_payload(payload)


def main() -> None:
    payload = json.load(sys.stdin)
    result = solve_payload(payload)
    print("CUOPT_RESULT_JSON:" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
