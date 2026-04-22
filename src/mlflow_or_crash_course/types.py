from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _coerce_int_tuple(values: list[Any] | tuple[Any, ...], field_name: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class SolverConfig:
    name: str
    library: str
    params: dict[str, Any]


@dataclass(frozen=True)
class PortfolioScenario:
    scenario_id: str
    values: tuple[int, ...]
    costs: tuple[int, ...]
    hours: tuple[int, ...]
    budget: int
    capacity: int
    project_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        item_count = len(self.values)
        if item_count == 0:
            raise ValueError("PortfolioScenario must contain at least one candidate project")
        if len(self.costs) != item_count or len(self.hours) != item_count:
            raise ValueError("values, costs, and hours must all have the same length")
        if self.budget <= 0 or self.capacity <= 0:
            raise ValueError("budget and capacity must both be positive")
        if any(value <= 0 for value in self.values):
            raise ValueError("values must all be positive")
        if any(cost <= 0 for cost in self.costs):
            raise ValueError("costs must all be positive")
        if any(hour <= 0 for hour in self.hours):
            raise ValueError("hours must all be positive")

        if not self.project_ids:
            object.__setattr__(
                self,
                "project_ids",
                tuple(f"P{index + 1:03d}" for index in range(item_count)),
            )
        elif len(self.project_ids) != item_count:
            raise ValueError("project_ids must match the number of decision variables")

    @property
    def item_count(self) -> int:
        return len(self.values)

    def to_request_record(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "project_ids_json": json.dumps(list(self.project_ids)),
            "values_json": json.dumps(list(self.values)),
            "costs_json": json.dumps(list(self.costs)),
            "hours_json": json.dumps(list(self.hours)),
            "budget": self.budget,
            "capacity": self.capacity,
        }

    @classmethod
    def from_lists(
        cls,
        *,
        scenario_id: str,
        values: list[Any] | tuple[Any, ...],
        costs: list[Any] | tuple[Any, ...],
        hours: list[Any] | tuple[Any, ...],
        budget: Any,
        capacity: Any,
        project_ids: list[Any] | tuple[Any, ...] | None = None,
    ) -> "PortfolioScenario":
        return cls(
            scenario_id=str(scenario_id),
            values=_coerce_int_tuple(values, "values"),
            costs=_coerce_int_tuple(costs, "costs"),
            hours=_coerce_int_tuple(hours, "hours"),
            budget=int(budget),
            capacity=int(capacity),
            project_ids=tuple(str(project_id) for project_id in project_ids or ()),
        )


@dataclass(frozen=True)
class PortfolioSolution:
    scenario_id: str
    library: str
    config_name: str
    status: str
    objective_value: float
    solve_time_ms: float
    total_cost: float
    total_hours: float
    selected_indices: tuple[int, ...]
    selected_project_ids: tuple[str, ...]
    is_feasible: bool
    is_optimal: bool

    @property
    def selected_count(self) -> int:
        return len(self.selected_indices)

    def to_record(self, *, budget: int, capacity: int) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "library": self.library,
            "config_name": self.config_name,
            "status": self.status,
            "objective_value": self.objective_value,
            "solve_time_ms": self.solve_time_ms,
            "total_cost": self.total_cost,
            "total_hours": self.total_hours,
            "selected_count": self.selected_count,
            "is_feasible": int(self.is_feasible),
            "is_optimal": int(self.is_optimal),
            "budget_utilization": self.total_cost / budget,
            "capacity_utilization": self.total_hours / capacity,
            "selected_indices_json": json.dumps(list(self.selected_indices)),
            "selected_project_ids_json": json.dumps(list(self.selected_project_ids)),
        }
