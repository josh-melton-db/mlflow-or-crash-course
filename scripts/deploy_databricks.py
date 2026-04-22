from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Sequence

from mlflow_or_crash_course.deployment import deploy_serving_endpoint


def _run(command: list[str]) -> None:
    print("$", " ".join(command))
    env = os.environ.copy()
    if command[:2] == ["databricks", "bundle"]:
        env.setdefault("DATABRICKS_BUNDLE_ENGINE", "direct")
    subprocess.run(command, check=True, env=env)


def _bundle_command(
    *,
    action: str,
    profile: str,
    target: str,
    variables: dict[str, str],
    resource_key: str | None = None,
) -> list[str]:
    command = ["databricks", "bundle", action]
    if resource_key:
        command.append(resource_key)
    command.extend(["-t", target, "--profile", profile])
    for key, value in variables.items():
        command.extend(["--var", f"{key}={value}"])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy the OR crash course workflow to Databricks.")
    parser.add_argument("--profile", default="azure")
    parser.add_argument("--target", default="azure")
    parser.add_argument("--catalog", default="main")
    parser.add_argument("--schema", default="or_blog_josh_melton")
    parser.add_argument(
        "--experiment-name",
        default="/Users/josh.melton@databricks.com/mlflow-or-crash-course",
    )
    parser.add_argument(
        "--registered-model-name",
        default="main.or_blog_josh_melton.portfolio_optimizer",
    )
    parser.add_argument("--scenario-count", default="6")
    parser.add_argument("--seed", default="7")
    parser.add_argument("--endpoint-name", default="portfolio-optimizer-endpoint")
    parser.add_argument("--workload-size", default="Small")
    parser.add_argument("--alias", default="Champion")
    parser.add_argument("--disable-scale-to-zero", action="store_true")
    parser.add_argument("--auto-capture-catalog")
    parser.add_argument("--auto-capture-schema")
    parser.add_argument("--auto-capture-table-prefix", default="or_crash_course")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    bundle_vars = {
        "catalog": args.catalog,
        "schema": args.schema,
        "experiment_name": args.experiment_name,
        "registered_model_name": args.registered_model_name,
        "scenario_count": args.scenario_count,
        "seed": args.seed,
    }

    _run(
        _bundle_command(
            action="deploy",
            profile=args.profile,
            target=args.target,
            variables=bundle_vars,
        )
    )
    _run(
        _bundle_command(
            action="run",
            profile=args.profile,
            target=args.target,
            resource_key="benchmark_or_solvers",
            variables=bundle_vars,
        )
    )

    endpoint_result = deploy_serving_endpoint(
        profile=args.profile,
        endpoint_name=args.endpoint_name,
        registered_model_name=args.registered_model_name,
        alias=args.alias,
        workload_size=args.workload_size,
        scale_to_zero_enabled=not args.disable_scale_to_zero,
        auto_capture_catalog=args.auto_capture_catalog,
        auto_capture_schema=args.auto_capture_schema,
        auto_capture_table_prefix=args.auto_capture_table_prefix,
    )
    print(json.dumps(endpoint_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
