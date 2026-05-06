"""Knapsack Solver — OR-Tools CP-SAT wrapped as an MLflow pyfunc.

Edit this file directly, then log it from the notebook.
Different configurations (time limits, workers) are passed via model_config
at log time — no need to modify this file for each experiment run.
"""

import pandas as pd
import numpy as np
import mlflow


class KnapsackSolver(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc that solves a 0/1 knapsack problem using OR-Tools CP-SAT."""

    def __init__(self, time_limit_s=5.0, num_workers=1):
        self.time_limit_s = time_limit_s
        self.num_workers = num_workers

    def predict(self, context, model_input, params=None):
        from ortools.sat.python import cp_model
        from time import perf_counter

        # model_input: DataFrame with columns [value, weight, capacity]
        # Each row is an item; capacity is constant across rows.
        values = model_input["value"].to_numpy(dtype=int)
        weights = model_input["weight"].to_numpy(dtype=int)
        capacity = int(model_input["capacity"].iloc[0])
        n = len(values)

        # Build CP-SAT model
        model = cp_model.CpModel()
        picks = [model.new_bool_var(f"x_{i}") for i in range(n)]

        # Capacity constraint
        model.add(sum(weights[i] * picks[i] for i in range(n)) <= capacity)

        # Objective: maximize total value
        model.maximize(sum(int(values[i]) * picks[i] for i in range(n)))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.num_search_workers = self.num_workers

        start = perf_counter()
        status = solver.solve(model)
        solve_ms = (perf_counter() - start) * 1000

        selected = [solver.value(picks[i]) for i in range(n)]
        total_value = sum(values[i] * selected[i] for i in range(n))
        total_weight = sum(weights[i] * selected[i] for i in range(n))

        return pd.DataFrame([{
            "selected": selected,
            "total_value": int(total_value),
            "total_weight": int(total_weight),
            "solve_time_ms": round(solve_ms, 2),
            "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        }])


# Required: tells MLflow which class is the model when logging as code
mlflow.models.set_model(KnapsackSolver())
