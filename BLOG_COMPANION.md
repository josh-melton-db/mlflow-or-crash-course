# Blog Companion

## Working Title

Inventory Optimization with MLflow on Databricks: Benchmark OR-Tools and SciPy, Promote a Champion, and Serve It

## Reader Promise

By the end of the post, the reader should understand how to:

- frame an inventory replenishment problem as an optimization workflow
- compare solver libraries and parameter settings with MLflow
- pick a winning configuration using operational metrics
- package the winner as an MLflow Model From Code
- deploy the result on Databricks serverless infrastructure

## Story Arc

1. Start with the idea that operations research experiments deserve the same rigor as machine learning experiments.
2. Introduce a weekly distribution-center replenishment problem with budget and storage constraints.
3. Show a small example scenario so the reader understands the decision variables and business objective.
4. Expand to a benchmark sweep across `OR-Tools CP-SAT` and `SciPy milp`.
5. Use MLflow to compare runs and explain why one configuration wins.
6. Register the champion as a custom model from code.
7. End with a live serving endpoint that returns replenishment recommendations from JSON input.

## Suggested Screenshots

- Notebook widgets and the opening markdown cell
- Small example scenario input table
- Recommended orders output for the small scenario
- MLflow comparison table for solver runs
- Registered model version or alias view
- Serving endpoint `READY` state
- Sample endpoint response with recommended orders JSON

## Points to Emphasize

- The example is integer replenishment planning, not a toy knapsack dressed up as supply chain.
- The comparison is not just about objective value; fill rate and solve time matter.
- MLflow is useful here even though the artifact is an OR model, not a neural network.
- Models From Code keeps the serving logic readable and inspectable.
- Databricks serverless resources keep the workflow lightweight for readers to reproduce.

## Optional Closing Angle

Close by noting that the same pattern generalizes to:

- safety stock tuning
- multi-echelon inventory planning
- transportation and routing models
- production scheduling
