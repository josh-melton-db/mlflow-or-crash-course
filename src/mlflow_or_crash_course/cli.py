from __future__ import annotations

import argparse
import json
from typing import Sequence

from .deployment import deploy_serving_endpoint
from .serving import build_input_example
from .workflow import run_benchmark_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and deploy OR solver workflows with MLflow.")
    subparsers = parser.add_subparsers(dest="command")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run the benchmark workflow.")
    benchmark_parser.add_argument("--experiment-name", default="mlflow-or-crash-course-local")
    benchmark_parser.add_argument("--registered-model-name")
    benchmark_parser.add_argument("--tracking-uri", default="file:./mlruns")
    benchmark_parser.add_argument("--registry-uri", default="")
    benchmark_parser.add_argument("--seed", type=int, default=7)
    benchmark_parser.add_argument("--scenario-count", type=int, default=6)
    benchmark_parser.add_argument("--output-dir", default="build/generated")
    benchmark_parser.add_argument("--run-name")

    deploy_parser = subparsers.add_parser("deploy-endpoint", help="Create or update a Databricks serving endpoint.")
    deploy_parser.add_argument("--profile", required=True)
    deploy_parser.add_argument("--endpoint-name", required=True)
    deploy_parser.add_argument("--registered-model-name", required=True)
    deploy_parser.add_argument("--model-version")
    deploy_parser.add_argument("--alias", default="Champion")
    deploy_parser.add_argument("--workload-size", default="Small")
    deploy_parser.add_argument("--disable-scale-to-zero", action="store_true")
    deploy_parser.add_argument("--auto-capture-catalog")
    deploy_parser.add_argument("--auto-capture-schema")
    deploy_parser.add_argument("--auto-capture-table-prefix", default="or_crash_course")

    subparsers.add_parser("sample-request", help="Print a JSON request payload for the served model.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "benchmark":
        result = run_benchmark_workflow(
            experiment_name=args.experiment_name,
            registered_model_name=args.registered_model_name,
            tracking_uri=args.tracking_uri,
            registry_uri=args.registry_uri,
            seed=args.seed,
            scenario_count=args.scenario_count,
            output_dir=args.output_dir,
            run_name=args.run_name,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "deploy-endpoint":
        result = deploy_serving_endpoint(
            profile=args.profile,
            endpoint_name=args.endpoint_name,
            registered_model_name=args.registered_model_name,
            model_version=args.model_version,
            alias=args.alias,
            workload_size=args.workload_size,
            scale_to_zero_enabled=not args.disable_scale_to_zero,
            auto_capture_catalog=args.auto_capture_catalog,
            auto_capture_schema=args.auto_capture_schema,
            auto_capture_table_prefix=args.auto_capture_table_prefix,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "sample-request":
        print(json.dumps(build_input_example().to_dict(orient="records"), indent=2))
        return

    parser.print_help()
