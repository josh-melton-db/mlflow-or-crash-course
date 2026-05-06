# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # OR-Ops Basics: Knapsack Solver with MLflow
# MAGIC
# MAGIC The simplest possible "OR-Ops" workflow:
# MAGIC
# MAGIC 1. Define a knapsack optimization problem
# MAGIC 2. Write a solver as a **pyfunc model** file
# MAGIC 3. Log two solver configurations to MLflow (different parameters)
# MAGIC 4. Register the best one to Unity Catalog
# MAGIC 5. Load the registered model and predict

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q ortools mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

mlflow.set_registry_uri("databricks-uc")

# ---------- Config ----------
CATALOG = "demos"
SCHEMA = "default"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.knapsack_solver"

current_user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT = f"/Users/{current_user}/or-ops-basics"
mlflow.set_experiment(EXPERIMENT)
print(f"Experiment: {EXPERIMENT}\nModel: {MODEL_NAME}")

# COMMAND ----------

# DBTITLE 1,Problem definition header
# MAGIC %md
# MAGIC ## 1. Define the Knapsack Problem
# MAGIC
# MAGIC Given items with weights and values, pick items to maximize total value without exceeding a weight capacity. This is the "hello world" of operations research.

# COMMAND ----------

# DBTITLE 1,Knapsack data
# A small knapsack instance (10 items)
items = pd.DataFrame({
    "item": [f"item_{i}" for i in range(10)],
    "value": [60, 100, 120, 80, 50, 70, 90, 110, 40, 95],
    "weight": [10, 20, 30, 15, 8, 12, 25, 28, 5, 18],
})
capacity = 60

print(f"Capacity: {capacity}")
display(items)

# COMMAND ----------

# DBTITLE 1,Solver model header
# MAGIC %md
# MAGIC ## 2. The Solver Lives in Its Own Python File
# MAGIC
# MAGIC The solver is a standalone `.py` file you can edit directly in the workspace editor:
# MAGIC
# MAGIC **`knapsack_solver.py`** (same folder as this notebook)
# MAGIC
# MAGIC It wraps OR-Tools CP-SAT as an `mlflow.pyfunc.PythonModel`. Different configurations (time limits, workers) are passed via `model_config` at log time — no need to modify the file for each experiment.

# COMMAND ----------

# DBTITLE 1,Solver pyfunc code
# The solver lives in knapsack_solver.py — same folder as this notebook
solver_file = Path.cwd() / "knapsack_solver.py"
assert solver_file.exists(), f"Solver file not found at {solver_file}"

print(f"Solver path: {solver_file}\n")
print(solver_file.read_text())

# COMMAND ----------

# DBTITLE 1,MLflow logging header
# MAGIC %md
# MAGIC ## 3. Log Two Solver Configurations to MLflow
# MAGIC
# MAGIC We log the same solver with different parameters — simulating an OR-Ops benchmark where you compare configurations.

# COMMAND ----------

# DBTITLE 1,Log two solver runs
from mlflow.models import infer_signature

# Prepare the input example (items + capacity column)
input_example = items.copy()
input_example["capacity"] = capacity

# Run the solver locally to get a sample output for the signature
import sys
sys.path.insert(0, str(solver_file.parent))
from knapsack_solver import KnapsackSolver

solver_local = KnapsackSolver(time_limit_s=5.0, num_workers=1)
result_sample = solver_local.predict(None, input_example)
signature = infer_signature(input_example, result_sample)

# --- Run 1: Baseline (1 worker, 5s time limit) ---
with mlflow.start_run(run_name="cpsat_1worker_5s") as run1:
    mlflow.log_params({"solver": "ortools_cpsat", "time_limit_s": 5.0, "num_workers": 1})
    result_1 = KnapsackSolver(time_limit_s=5.0, num_workers=1).predict(None, input_example)
    mlflow.log_metrics({
        "total_value": result_1["total_value"].iloc[0],
        "total_weight": result_1["total_weight"].iloc[0],
        "solve_time_ms": result_1["solve_time_ms"].iloc[0],
    })
    model_info_1 = mlflow.pyfunc.log_model(
        name="knapsack_solver",
        python_model=str(solver_file),
        signature=signature,
        input_example=input_example,
        pip_requirements=["ortools", "pandas", "numpy"],
        model_config={"time_limit_s": 5.0, "num_workers": 1},
    )
    print(f"Run 1: value={result_1['total_value'].iloc[0]}, time={result_1['solve_time_ms'].iloc[0]:.2f}ms")

# --- Run 2: More workers, tighter time limit ---
with mlflow.start_run(run_name="cpsat_4workers_2s") as run2:
    mlflow.log_params({"solver": "ortools_cpsat", "time_limit_s": 2.0, "num_workers": 4})
    result_2 = KnapsackSolver(time_limit_s=2.0, num_workers=4).predict(None, input_example)
    mlflow.log_metrics({
        "total_value": result_2["total_value"].iloc[0],
        "total_weight": result_2["total_weight"].iloc[0],
        "solve_time_ms": result_2["solve_time_ms"].iloc[0],
    })
    model_info_2 = mlflow.pyfunc.log_model(
        name="knapsack_solver",
        python_model=str(solver_file),
        signature=signature,
        input_example=input_example,
        pip_requirements=["ortools", "pandas", "numpy"],
        model_config={"time_limit_s": 2.0, "num_workers": 4},
    )
    print(f"Run 2: value={result_2['total_value'].iloc[0]}, time={result_2['solve_time_ms'].iloc[0]:.2f}ms")

# COMMAND ----------

# DBTITLE 1,Registration header
# MAGIC %md
# MAGIC ## 4. Register the Best Configuration

# COMMAND ----------

# DBTITLE 1,Register the best model
# Compare: pick the run with the highest objective (total_value), break ties by speed
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT],
    order_by=["metrics.total_value DESC", "metrics.solve_time_ms ASC"],
    max_results=2,
)

# Show available metrics columns
cols = [c for c in 
        ["run_id", "run_name", "tags.mlflow.runName", "params.num_workers", 
         "metrics.total_value", "metrics.solve_time_ms"] 
        if c in runs.columns]
display(runs[cols])

# Register the best
best_run_id = runs.iloc[0]["run_id"]
best_model_uri = f"runs:/{best_run_id}/knapsack_solver"

registered_version = mlflow.register_model(
    model_uri=best_model_uri,
    name=MODEL_NAME,
)
print(f"\nRegistered: {MODEL_NAME} version {registered_version.version}")

# COMMAND ----------

# DBTITLE 1,Predict header
# MAGIC %md
# MAGIC ## 5. Predict with the Registered Model

# COMMAND ----------

# DBTITLE 1,Predict with registered model
# Load the registered model and solve a new problem
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{registered_version.version}")

# New problem instance — different items, different capacity
new_items = pd.DataFrame({
    "item": ["laptop", "camera", "book", "phone", "tablet"],
    "value": [500, 300, 50, 400, 350],
    "weight": [40, 25, 5, 15, 20],
    "capacity": [55] * 5,  # capacity repeated per row
})

prediction = model.predict(new_items)
print("\n--- Solution ---")
print(f"Selected items: {[new_items['item'].iloc[i] for i, s in enumerate(prediction['selected'].iloc[0]) if s]}")
print(f"Total value: {prediction['total_value'].iloc[0]}")
print(f"Total weight: {prediction['total_weight'].iloc[0]} / {55}")
print(f"Solve time: {prediction['solve_time_ms'].iloc[0]:.2f} ms")
