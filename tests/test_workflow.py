from pathlib import Path

from mlflow_or_crash_course.serving import build_input_example, scenario_from_request_row
from mlflow_or_crash_course.workflow import _render_model_script, run_benchmark_workflow


def test_request_example_round_trips() -> None:
    row = build_input_example().to_dict(orient="records")[0]
    scenario = scenario_from_request_row(row)

    assert scenario.scenario_id == "example_capital_plan"
    assert scenario.budget == 95
    assert scenario.capacity == 48
    assert len(scenario.project_ids) == 5


def test_workflow_runs_without_registration(tmp_path: Path) -> None:
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    result = run_benchmark_workflow(
        experiment_name="unit-test-experiment",
        tracking_uri=tracking_uri,
        registry_uri="",
        seed=13,
        scenario_count=3,
        output_dir=str(tmp_path / "generated"),
    )

    assert result["run_id"]
    assert result["registered_model_version"] is None
    assert result["champion"]["config_name"] in {
        "ortools_single_thread",
        "ortools_parallel",
        "scipy_fast_gap",
        "scipy_exact",
    }


def test_model_script_drops_nan_parameters(tmp_path: Path) -> None:
    champion = {
        "library": "ortools_cp_sat",
        "config_name": "ortools_single_thread",
        "param__time_limit_s": 3.0,
        "param__num_workers": 1.0,
        "param__relative_gap": 0.0,
        "param__mip_rel_gap": float("nan"),
        "param__presolve": float("nan"),
    }
    script_path = _render_model_script(champion, tmp_path / "champion_model.py")
    contents = script_path.read_text(encoding="utf-8")

    assert "nan" not in contents
    assert "'num_workers': 1.0" in contents
