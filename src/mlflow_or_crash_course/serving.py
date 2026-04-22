from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .problem import example_scenario
from .types import PortfolioScenario, PortfolioSolution


def _coerce_sequence(value: Any, field_name: str) -> list[Any]:
    if value is None:
        raise ValueError(f"Missing required field: {field_name}")
    if isinstance(value, str):
        return list(json.loads(value))
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    raise TypeError(f"Unsupported payload type for {field_name}: {type(value)!r}")


def scenario_from_request_row(row: dict[str, Any]) -> PortfolioScenario:
    project_ids = _coerce_sequence(row.get("project_ids_json"), "project_ids_json")
    values = _coerce_sequence(row.get("values_json"), "values_json")
    costs = _coerce_sequence(row.get("costs_json"), "costs_json")
    hours = _coerce_sequence(row.get("hours_json"), "hours_json")

    return PortfolioScenario.from_lists(
        scenario_id=str(row.get("scenario_id") or "serving_request"),
        project_ids=project_ids,
        values=values,
        costs=costs,
        hours=hours,
        budget=row.get("budget"),
        capacity=row.get("capacity"),
    )


def solution_to_response_record(solution: PortfolioSolution) -> dict[str, Any]:
    return {
        "scenario_id": solution.scenario_id,
        "library": solution.library,
        "config_name": solution.config_name,
        "status": solution.status,
        "is_feasible": solution.is_feasible,
        "is_optimal": solution.is_optimal,
        "objective_value": solution.objective_value,
        "solve_time_ms": solution.solve_time_ms,
        "total_cost": solution.total_cost,
        "total_hours": solution.total_hours,
        "selected_count": solution.selected_count,
        "selected_indices_json": json.dumps(list(solution.selected_indices)),
        "selected_project_ids_json": json.dumps(list(solution.selected_project_ids)),
    }


def build_input_example() -> pd.DataFrame:
    return pd.DataFrame([example_scenario().to_request_record()])
