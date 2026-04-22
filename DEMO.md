# Demo Overview

## Goal

Build a concise, blog-friendly demo that shows how to experiment with an operations research model using MLflow, compare solver libraries and parameter settings, promote the winner, and deploy it on Databricks serverless infrastructure.

## Audience

- Data scientists who already use MLflow and want a non-ML example
- Solution architects who need a practical OR + Databricks storyline
- Readers who want a minimal but real end-to-end deployment example

## Business Story

The demo frames optimization as a capital allocation problem:

- each project has an expected value
- each project consumes budget
- each project consumes implementation hours
- the optimizer chooses the best subset under those constraints

This keeps the math simple enough for a blog post while still feeling like a real planning problem.

## What the demo highlights

1. `OR-Tools CP-SAT` vs `SciPy milp`
2. parameter sweeps across both libraries
3. MLflow experiment tracking for solver runs
4. champion selection based on objective value and solve time
5. MLflow Models From Code for a custom optimization model
6. Databricks serverless job execution
7. Unity Catalog model registration
8. Databricks Model Serving deployment

## Architecture

1. Local package code defines the optimization problem, solver wrappers, benchmark workflow, and deployment helpers.
2. A Databricks bundle deploys:
   - a Unity Catalog schema
   - a registered model shell
   - a serverless benchmark job
3. The serverless job runs the benchmark, logs results to MLflow, and registers the champion model version.
4. A deployment helper updates a serving endpoint to point to the `Champion` model alias.

## Design principles

- Keep the optimization problem small enough to explain in one post
- Prefer serverless resources everywhere
- Use MLflow-native logging and registration
- Keep the served model thin and readable
- Make the repo runnable locally before deploying remotely
