from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from .problem import build_request_rows, generate_benchmark_scenarios
from .serving import build_input_example
from .solvers import solve_scenario
from .types import PortfolioScenario, SolverConfig

DEFAULT_SOLVER_CONFIGS = [
    SolverConfig(
        name="ortools_single_thread",
        library="ortools_cp_sat",
        params={"time_limit_s": 3.0, "num_workers": 1, "relative_gap": 0.0},
    ),
    SolverConfig(
        name="ortools_parallel",
        library="ortools_cp_sat",
        params={"time_limit_s": 3.0, "num_workers": 8, "relative_gap": 0.0},
    ),
    SolverConfig(
        name="scipy_fast_gap",
        library="scipy_milp",
        params={"time_limit_s": 3.0, "mip_rel_gap": 0.02, "presolve": True},
    ),
    SolverConfig(
        name="scipy_exact",
        library="scipy_milp",
        params={"time_limit_s": 6.0, "mip_rel_gap": 0.0, "presolve": True},
    ),
]


def _benchmark_single_config(
    scenarios: list[PortfolioScenario],
    config: SolverConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for scenario in scenarios:
        solution = solve_scenario(
            scenario,
            library=config.library,
            config_name=config.name,
            solver_params=config.params,
        )
        record = solution.to_record(budget=scenario.budget, capacity=scenario.capacity)
        record.update(
            {
                "item_count": scenario.item_count,
                "budget": scenario.budget,
                "capacity": scenario.capacity,
            }
        )
        records.append(record)

    frame = pd.DataFrame(records)
    feasible = frame[frame["is_feasible"] == 1]
    summary = {
        "config_name": config.name,
        "library": config.library,
        "avg_objective": float(frame["objective_value"].mean()),
        "avg_solve_time_ms": float(frame["solve_time_ms"].mean()),
        "feasible_ratio": float(frame["is_feasible"].mean()),
        "optimal_ratio": float(frame["is_optimal"].mean()),
        "avg_selected_count": float(frame["selected_count"].mean()),
        "avg_budget_utilization": float(feasible["budget_utilization"].mean()) if not feasible.empty else 0.0,
        "avg_capacity_utilization": float(feasible["capacity_utilization"].mean()) if not feasible.empty else 0.0,
        "scenario_count": int(len(frame)),
    }
    for key, value in config.params.items():
        summary[f"param__{key}"] = value
    return frame, summary


def _select_champion(summary_frame: pd.DataFrame) -> dict[str, Any]:
    ordered = summary_frame.sort_values(
        by=["feasible_ratio", "avg_objective", "optimal_ratio", "avg_solve_time_ms"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    champion = ordered.iloc[0].to_dict()
    champion["selection_rule"] = "max feasible_ratio, max avg_objective, max optimal_ratio, min avg_solve_time_ms"
    champion["rank"] = 1
    return champion


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_requirements(champion_library: str) -> list[str]:
    packages = ["mlflow", "pandas"]
    if champion_library == "ortools_cp_sat":
        packages.append("ortools")
    elif champion_library == "scipy_milp":
        packages.extend(["numpy", "scipy"])
    return [f"{package}=={version(package)}" for package in packages]


def _render_model_script(champion: dict[str, Any], output_path: Path) -> Path:
    params_repr = repr(
        {
            key.replace("param__", ""): value
            for key, value in champion.items()
            if key.startswith("param__") and value is not None and not pd.isna(value)
        }
    )
    library = champion["library"]
    config_name = champion["config_name"]

    if library == "ortools_cp_sat":
        script = f"""
import json
from typing import Any

import pandas as pd
from mlflow.models import set_model
from mlflow.pyfunc import PythonModel
from ortools.sat.python import cp_model

CHAMPION_LIBRARY = {library!r}
CHAMPION_CONFIG_NAME = {config_name!r}
CHAMPION_PARAMS = {params_repr}


def _coerce_sequence(value: Any, field_name: str) -> list[Any]:
    if value is None:
        raise ValueError(f"Missing required field: {{field_name}}")
    if isinstance(value, str):
        return list(json.loads(value))
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    raise TypeError(f"Unsupported payload type for {{field_name}}: {{type(value)!r}}")


def _solve_row(row: dict[str, Any]) -> dict[str, Any]:
    project_ids = [str(project_id) for project_id in _coerce_sequence(row.get("project_ids_json"), "project_ids_json")]
    values = [int(value) for value in _coerce_sequence(row.get("values_json"), "values_json")]
    costs = [int(cost) for cost in _coerce_sequence(row.get("costs_json"), "costs_json")]
    hours = [int(hour) for hour in _coerce_sequence(row.get("hours_json"), "hours_json")]
    budget = int(row.get("budget"))
    capacity = int(row.get("capacity"))

    model = cp_model.CpModel()
    variables = [model.NewBoolVar(project_id) for project_id in project_ids]
    model.Add(sum(cost * var for cost, var in zip(costs, variables)) <= budget)
    model.Add(sum(hour * var for hour, var in zip(hours, variables)) <= capacity)
    model.Maximize(sum(value * var for value, var in zip(values, variables)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(CHAMPION_PARAMS.get("time_limit_s", 3.0))
    solver.parameters.num_search_workers = int(CHAMPION_PARAMS.get("num_workers", 8))
    solver.parameters.relative_gap_limit = float(CHAMPION_PARAMS.get("relative_gap", 0.0))
    status_code = solver.Solve(model)

    status_lookup = {{
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }}
    is_feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    selected_indices = [index for index, variable in enumerate(variables) if is_feasible and solver.BooleanValue(variable)]

    return {{
        "scenario_id": str(row.get("scenario_id") or "serving_request"),
        "library": CHAMPION_LIBRARY,
        "config_name": CHAMPION_CONFIG_NAME,
        "status": status_lookup.get(status_code, f"STATUS_{{status_code}}"),
        "is_feasible": bool(is_feasible),
        "is_optimal": bool(status_code == cp_model.OPTIMAL),
        "objective_value": float(sum(values[index] for index in selected_indices)),
        "total_cost": float(sum(costs[index] for index in selected_indices)),
        "total_hours": float(sum(hours[index] for index in selected_indices)),
        "selected_count": len(selected_indices),
        "selected_indices_json": json.dumps(selected_indices),
        "selected_project_ids_json": json.dumps([project_ids[index] for index in selected_indices]),
    }}


class PortfolioOptimizerModel(PythonModel):
    def predict(self, context, model_input, params=None):
        frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        return pd.DataFrame([_solve_row(record) for record in frame.to_dict(orient="records")])


set_model(PortfolioOptimizerModel())
"""
    elif library == "scipy_milp":
        script = f"""
import json
from typing import Any

import numpy as np
import pandas as pd
from mlflow.models import set_model
from mlflow.pyfunc import PythonModel
from scipy.optimize import Bounds, LinearConstraint, milp

CHAMPION_LIBRARY = {library!r}
CHAMPION_CONFIG_NAME = {config_name!r}
CHAMPION_PARAMS = {params_repr}


def _coerce_sequence(value: Any, field_name: str) -> list[Any]:
    if value is None:
        raise ValueError(f"Missing required field: {{field_name}}")
    if isinstance(value, str):
        return list(json.loads(value))
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    raise TypeError(f"Unsupported payload type for {{field_name}}: {{type(value)!r}}")


def _solve_row(row: dict[str, Any]) -> dict[str, Any]:
    project_ids = [str(project_id) for project_id in _coerce_sequence(row.get("project_ids_json"), "project_ids_json")]
    values = [int(value) for value in _coerce_sequence(row.get("values_json"), "values_json")]
    costs = [int(cost) for cost in _coerce_sequence(row.get("costs_json"), "costs_json")]
    hours = [int(hour) for hour in _coerce_sequence(row.get("hours_json"), "hours_json")]
    budget = int(row.get("budget"))
    capacity = int(row.get("capacity"))

    options = {{
        "time_limit": float(CHAMPION_PARAMS.get("time_limit_s", 3.0)),
        "mip_rel_gap": float(CHAMPION_PARAMS.get("mip_rel_gap", 0.0)),
        "presolve": bool(CHAMPION_PARAMS.get("presolve", True)),
    }}

    result = milp(
        c=-np.asarray(values, dtype=float),
        constraints=LinearConstraint(
            np.asarray([costs, hours], dtype=float),
            np.asarray([-np.inf, -np.inf], dtype=float),
            np.asarray([budget, capacity], dtype=float),
        ),
        integrality=np.ones(len(values), dtype=int),
        bounds=Bounds(np.zeros(len(values)), np.ones(len(values))),
        options=options,
    )

    status_lookup = {{
        0: "OPTIMAL",
        1: "LIMIT_REACHED",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "OTHER",
    }}
    selected_indices = []
    is_feasible = False
    if result.x is not None:
        rounded = np.rint(result.x).astype(int)
        selected_indices = [index for index, include in enumerate(rounded.tolist()) if include == 1]
        is_feasible = (
            sum(costs[index] for index in selected_indices) <= budget
            and sum(hours[index] for index in selected_indices) <= capacity
        )
        if not is_feasible:
            selected_indices = []

    return {{
        "scenario_id": str(row.get("scenario_id") or "serving_request"),
        "library": CHAMPION_LIBRARY,
        "config_name": CHAMPION_CONFIG_NAME,
        "status": status_lookup.get(result.status, f"STATUS_{{result.status}}"),
        "is_feasible": bool(is_feasible),
        "is_optimal": bool(result.status == 0 and is_feasible),
        "objective_value": float(sum(values[index] for index in selected_indices)),
        "total_cost": float(sum(costs[index] for index in selected_indices)),
        "total_hours": float(sum(hours[index] for index in selected_indices)),
        "selected_count": len(selected_indices),
        "selected_indices_json": json.dumps(selected_indices),
        "selected_project_ids_json": json.dumps([project_ids[index] for index in selected_indices]),
    }}


class PortfolioOptimizerModel(PythonModel):
    def predict(self, context, model_input, params=None):
        frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        return pd.DataFrame([_solve_row(record) for record in frame.to_dict(orient="records")])


set_model(PortfolioOptimizerModel())
"""
    else:
        raise ValueError(f"Unsupported champion library: {library}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(textwrap.dedent(script).strip() + "\n", encoding="utf-8")
    return output_path


def _resolve_registered_model_version(model_name: str, run_id: str) -> str | None:
    client = MlflowClient()
    matching_versions = [
        version_info
        for version_info in client.search_model_versions(f"name='{model_name}'")
        if version_info.run_id == run_id
    ]
    if not matching_versions:
        return None
    latest = max(matching_versions, key=lambda version_info: int(version_info.version))
    client.set_registered_model_alias(model_name, "Champion", latest.version)
    return str(latest.version)


def run_benchmark_workflow(
    *,
    experiment_name: str,
    registered_model_name: str | None = None,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
    seed: int = 7,
    scenario_count: int = 6,
    output_dir: str = "build/generated",
    run_name: str | None = None,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(experiment_name)

    scenarios = generate_benchmark_scenarios(seed=seed, scenario_count=scenario_count)
    run_name = run_name or f"portfolio_benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    output_path = Path(output_dir)

    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        summary_rows: list[dict[str, Any]] = []
        request_examples = pd.DataFrame(build_request_rows(scenarios[: min(3, len(scenarios))]))

        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.log_params(
                {
                    "seed": seed,
                    "scenario_count": scenario_count,
                    "solver_config_count": len(DEFAULT_SOLVER_CONFIGS),
                    "tracking_uri": tracking_uri,
                    "registry_uri": registry_uri,
                    "registered_model_name": registered_model_name or "not_registered",
                }
            )

            requests_path = temp_root / "benchmark_request_examples.csv"
            _write_dataframe(requests_path, request_examples)
            mlflow.log_artifact(str(requests_path), artifact_path="benchmark")

            for config in DEFAULT_SOLVER_CONFIGS:
                config_frame, summary = _benchmark_single_config(scenarios, config)
                summary_rows.append(summary)

                with mlflow.start_run(run_name=config.name, nested=True):
                    mlflow.log_param("library", config.library)
                    for key, value in config.params.items():
                        mlflow.log_param(f"solver__{key}", value)
                    mlflow.log_metrics(
                        {
                            "avg_objective": summary["avg_objective"],
                            "avg_solve_time_ms": summary["avg_solve_time_ms"],
                            "feasible_ratio": summary["feasible_ratio"],
                            "optimal_ratio": summary["optimal_ratio"],
                            "avg_selected_count": summary["avg_selected_count"],
                        }
                    )
                    config_path = temp_root / f"{config.name}_scenario_results.csv"
                    _write_dataframe(config_path, config_frame)
                    mlflow.log_artifact(str(config_path), artifact_path="benchmark")

            summary_frame = pd.DataFrame(summary_rows)
            champion = _select_champion(summary_frame)
            champion_config = next(
                config for config in DEFAULT_SOLVER_CONFIGS if config.name == champion["config_name"]
            )

            summary_path = temp_root / "benchmark_summary.csv"
            _write_dataframe(summary_path, summary_frame.sort_values("avg_solve_time_ms"))
            mlflow.log_artifact(str(summary_path), artifact_path="benchmark")
            mlflow.log_dict(champion, "benchmark/champion.json")
            mlflow.log_metrics(
                {
                    "champion_avg_objective": float(champion["avg_objective"]),
                    "champion_avg_solve_time_ms": float(champion["avg_solve_time_ms"]),
                    "champion_feasible_ratio": float(champion["feasible_ratio"]),
                }
            )
            mlflow.set_tag("champion_config_name", str(champion["config_name"]))
            mlflow.set_tag("champion_library", str(champion["library"]))

            model_version: str | None = None
            model_info = None
            if registered_model_name:
                model_script_path = _render_model_script(champion, output_path / "champion_model.py")
                model_info = mlflow.pyfunc.log_model(
                    name="portfolio_optimizer",
                    python_model=str(model_script_path),
                    registered_model_name=registered_model_name,
                    input_example=build_input_example(),
                    pip_requirements=_build_requirements(champion["library"]),
                )
                model_version = _resolve_registered_model_version(
                    registered_model_name,
                    active_run.info.run_id,
                )

            return {
                "run_id": active_run.info.run_id,
                "experiment_name": experiment_name,
                "registered_model_name": registered_model_name,
                "registered_model_version": model_version,
                "model_uri": getattr(model_info, "model_uri", None) if model_info else None,
                "champion": champion,
                "champion_config": asdict(champion_config),
            }
