from mlflow_or_crash_course.problem import example_scenario
from mlflow_or_crash_course.solvers import solve_scenario


def test_ortools_and_scipy_match_on_example_scenario() -> None:
    scenario = example_scenario()

    ortools_solution = solve_scenario(
        scenario,
        library="ortools_cp_sat",
        config_name="ortools_test",
        solver_params={"time_limit_s": 2.0, "num_workers": 1, "relative_gap": 0.0},
    )
    scipy_solution = solve_scenario(
        scenario,
        library="scipy_milp",
        config_name="scipy_test",
        solver_params={"time_limit_s": 2.0, "mip_rel_gap": 0.0, "presolve": True},
    )

    assert ortools_solution.is_feasible
    assert scipy_solution.is_feasible
    assert ortools_solution.objective_value == 290.0
    assert scipy_solution.objective_value == 290.0
    assert set(ortools_solution.selected_project_ids) == {"warehouse", "route-optimizer"}
    assert set(scipy_solution.selected_project_ids) == {"warehouse", "route-optimizer"}
