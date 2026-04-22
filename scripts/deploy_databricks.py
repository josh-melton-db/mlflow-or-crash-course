from __future__ import annotations

import argparse
import os
import subprocess
from typing import Sequence


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
        if value != "":
            command.extend(["--var", f"{key}={value}"])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy the public inventory optimization notebook workflow to Databricks.")
    parser.add_argument("--profile", default="azure")
    parser.add_argument("--target", default="azure")
    parser.add_argument("--catalog", default="main")
    parser.add_argument("--schema", default="inventory_optimization_blog")
    parser.add_argument(
        "--experiment-name",
        default="",
    )
    parser.add_argument(
        "--registered-model-name",
        default="",
    )
    parser.add_argument("--endpoint-name", default="inventory-optimizer-endpoint")
    parser.add_argument("--scenario-count", default="6")
    parser.add_argument("--seed", default="7")
    parser.add_argument("--deploy-endpoint", choices=["true", "false"], default="true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    bundle_vars = {
        "catalog": args.catalog,
        "schema": args.schema,
        "experiment_name": args.experiment_name,
        "registered_model_name": args.registered_model_name,
        "endpoint_name": args.endpoint_name,
        "scenario_count": args.scenario_count,
        "seed": args.seed,
        "deploy_endpoint": args.deploy_endpoint,
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
            resource_key="inventory_optimization_crash_course",
            variables=bundle_vars,
        )
    )


if __name__ == "__main__":
    main()
