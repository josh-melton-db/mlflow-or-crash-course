from __future__ import annotations

import argparse
import json
from typing import Sequence

from .workflow import run_benchmark_workflow


def build_benchmark_job_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark OR solvers inside a Databricks job.")
    parser.add_argument("--experiment-name", "--experiment_name", dest="experiment_name", required=True)
    parser.add_argument(
        "--registered-model-name",
        "--registered_model_name",
        dest="registered_model_name",
        required=True,
    )
    parser.add_argument("--tracking-uri", "--tracking_uri", dest="tracking_uri", default="databricks")
    parser.add_argument("--registry-uri", "--registry_uri", dest="registry_uri", default="databricks-uc")
    parser.add_argument("--seed", default="7")
    parser.add_argument("--scenario-count", "--scenario_count", dest="scenario_count", default="6")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="/local_disk0/tmp/mlflow-or-crash-course",
    )
    parser.add_argument("--run-name", "--run_name", dest="run_name", default="")
    return parser


def benchmark_job(
    argv: Sequence[str] | None = None,
) -> None:
    args = build_benchmark_job_parser().parse_args(list(argv) if argv is not None else None)
    result = run_benchmark_workflow(
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        seed=int(args.seed),
        scenario_count=int(args.scenario_count),
        output_dir=args.output_dir,
        run_name=args.run_name or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
