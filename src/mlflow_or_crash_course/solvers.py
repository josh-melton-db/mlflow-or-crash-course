from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, milp

from .types import PortfolioScenario, PortfolioSolution


def _build_solution(
    *,
    scenario: PortfolioScenario,
    library: str,
    config_name: str,
    status: str,
    solve_time_ms: float,
    selected_indices: list[int],
    is_feasible: bool,
    is_optimal: bool,
) -> PortfolioSolution:
    total_cost = float(sum(scenario.costs[index] for index in selected_indices))
    total_hours = float(sum(scenario.hours[index] for index in selected_indices))
    objective_value = float(sum(scenario.values[index] for index in selected_indices))
    return PortfolioSolution(
        scenario_id=scenario.scenario_id,
        library=library,
        config_name=config_name,
        status=status,
        objective_value=objective_value,
        solve_time_ms=solve_time_ms,
        total_cost=total_cost,
        total_hours=total_hours,
        selected_indices=tuple(selected_indices),
        selected_project_ids=tuple(scenario.project_ids[index] for index in selected_indices),
        is_feasible=is_feasible,
        is_optimal=is_optimal,
    )


def _is_selection_feasible(scenario: PortfolioScenario, selected_indices: list[int]) -> bool:
    total_cost = sum(scenario.costs[index] for index in selected_indices)
    total_hours = sum(scenario.hours[index] for index in selected_indices)
    return total_cost <= scenario.budget and total_hours <= scenario.capacity


def solve_with_ortools(
    scenario: PortfolioScenario,
    *,
    config_name: str,
    time_limit_s: float = 3.0,
    num_workers: int = 8,
    relative_gap: float = 0.0,
) -> PortfolioSolution:
    model = cp_model.CpModel()
    variables = [model.NewBoolVar(project_id) for project_id in scenario.project_ids]

    model.Add(sum(cost * var for cost, var in zip(scenario.costs, variables)) <= scenario.budget)
    model.Add(sum(hour * var for hour, var in zip(scenario.hours, variables)) <= scenario.capacity)
    model.Maximize(sum(value * var for value, var in zip(scenario.values, variables)))

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
    selected_indices = (
        [index for index, var in enumerate(variables) if solver.BooleanValue(var)] if is_feasible else []
    )

    return _build_solution(
        scenario=scenario,
        library="ortools_cp_sat",
        config_name=config_name,
        status=status,
        solve_time_ms=solve_time_ms,
        selected_indices=selected_indices,
        is_feasible=is_feasible,
        is_optimal=status_code == cp_model.OPTIMAL,
    )


def solve_with_scipy(
    scenario: PortfolioScenario,
    *,
    config_name: str,
    time_limit_s: float = 3.0,
    mip_rel_gap: float = 0.0,
    presolve: bool = True,
    node_limit: int | None = None,
) -> PortfolioSolution:
    coefficients = -np.asarray(scenario.values, dtype=float)
    constraint_matrix = np.asarray([scenario.costs, scenario.hours], dtype=float)
    upper_bounds = np.asarray([scenario.budget, scenario.capacity], dtype=float)
    lower_bounds = np.asarray([-np.inf, -np.inf], dtype=float)

    options: dict[str, Any] = {
        "time_limit": float(time_limit_s),
        "mip_rel_gap": float(mip_rel_gap),
        "presolve": bool(presolve),
    }
    if node_limit is not None:
        options["node_limit"] = int(node_limit)

    started = perf_counter()
    result = milp(
        c=coefficients,
        constraints=LinearConstraint(constraint_matrix, lower_bounds, upper_bounds),
        integrality=np.ones(scenario.item_count, dtype=int),
        bounds=Bounds(np.zeros(scenario.item_count), np.ones(scenario.item_count)),
        options=options,
    )
    solve_time_ms = (perf_counter() - started) * 1000

    status_lookup = {
        0: "OPTIMAL",
        1: "LIMIT_REACHED",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "OTHER",
    }
    status = status_lookup.get(result.status, f"STATUS_{result.status}")
    selected_indices: list[int] = []
    is_feasible = False

    if result.x is not None:
        rounded = np.rint(result.x).astype(int)
        selected_indices = [index for index, include in enumerate(rounded.tolist()) if include == 1]
        is_feasible = _is_selection_feasible(scenario, selected_indices)

    return _build_solution(
        scenario=scenario,
        library="scipy_milp",
        config_name=config_name,
        status=status,
        solve_time_ms=solve_time_ms,
        selected_indices=selected_indices if is_feasible else [],
        is_feasible=is_feasible,
        is_optimal=result.status == 0 and is_feasible,
    )


def solve_scenario(
    scenario: PortfolioScenario,
    *,
    library: str,
    config_name: str,
    solver_params: dict[str, Any],
) -> PortfolioSolution:
    if library == "ortools_cp_sat":
        return solve_with_ortools(scenario, config_name=config_name, **solver_params)
    if library == "scipy_milp":
        return solve_with_scipy(scenario, config_name=config_name, **solver_params)
    raise ValueError(f"Unsupported solver library: {library}")
