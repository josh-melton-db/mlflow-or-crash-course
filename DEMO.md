# Demo Overview

## Goal

Build a concise, blog-friendly demo that shows how to experiment with an inventory optimization model using MLflow, compare solver libraries and parameter settings, promote the winner, and deploy it on Databricks serverless infrastructure from a single notebook.

## Audience

- Data scientists who already use MLflow and want a non-ML example
- Solution architects who need a practical OR + Databricks storyline
- Readers who want a minimal but real end-to-end deployment example

## Business Story

The demo frames optimization as a distribution-center replenishment problem:

- each SKU has on-hand inventory and forecast demand
- ordering cases consumes procurement budget
- ordering cases consumes storage capacity
- leftover inventory carries a holding cost
- stockouts incur a service penalty
- the optimizer chooses integer reorder quantities that maximize business value

This keeps the math simple enough for a blog post while still feeling like a real supply-chain planning problem.

## What the demo highlights

1. `OR-Tools CP-SAT` vs `SciPy milp`
2. parameter sweeps across both libraries
3. MLflow experiment tracking for solver runs
4. champion selection based on fill rate, objective value, and solve time
5. MLflow Models From Code for a custom optimization model
6. Databricks serverless job execution
7. Unity Catalog model registration
8. Databricks Model Serving deployment

## Architecture

1. A single Databricks source notebook defines the optimization problem, solver wrappers, benchmark workflow, and optional serving deployment.
2. A Databricks bundle deploys:
   - a Unity Catalog schema
   - a registered model shell
   - a serverless notebook job
3. The notebook job runs the benchmark, logs results to MLflow, registers the champion model version, and can create or update the serving endpoint.
4. A thin local deployment helper only deploys the bundle and runs the notebook.

## Design principles

- Keep the optimization problem small enough to explain in one post
- Make the notebook the main artifact, not a local Python package
- Prefer serverless resources everywhere
- Use MLflow-native logging and registration
- Keep the served model thin and readable
- Keep the local footprint minimal and deployment-oriented
