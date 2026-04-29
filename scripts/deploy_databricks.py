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
    parser = argparse.ArgumentParser(description="Deploy an inventory optimization notebook workflow to Databricks.")
    parser.add_argument("--profile", default="azure")
    parser.add_argument("--target", default="azure")
    parser.add_argument("--catalog", default="demos")
    parser.add_argument("--schema", default="default")
    parser.add_argument(
        "--experiment-name",
        default="",
    )
    parser.add_argument(
        "--model-name",
        default="inventory_optimization",
    )
    parser.add_argument("--endpoint-name", default="inventory-optimizer-endpoint")
    parser.add_argument("--gpu-model-name", default="inventory_optimization_cuopt")
    parser.add_argument("--gpu-endpoint-name", default="inventory-optimizer-cuopt-gpu-endpoint")
    parser.add_argument("--gpu-serving-workload-type", default="GPU_SMALL")
    parser.add_argument("--gpu-hardware-accelerator", default="GPU_1xA10")
    parser.add_argument("--gpu-environment-version", default="4")
    parser.add_argument("--scenario-count", default="6")
    parser.add_argument("--small-sku-counts", default="18;36;54;72")
    parser.add_argument("--seed", default="7")
    parser.add_argument("--deploy-endpoint", choices=["true", "false"], default="true")
    parser.add_argument("--run-large-benchmark", choices=["true", "false"], default="false")
    parser.add_argument("--large-product-count", default="80")
    parser.add_argument("--large-source-count", default="12")
    parser.add_argument("--large-dc-count", default="80")
    parser.add_argument("--large-store-count", default="250")
    parser.add_argument("--large-sources-per-dc", default="4")
    parser.add_argument("--large-dcs-per-store", default="4")
    parser.add_argument("--large-time-limit-s", default="600")
    parser.add_argument("--large-experiment-name", default="")
    parser.add_argument("--large-benchmark-id", default="large_default")
    parser.add_argument(
        "--resource-key",
        choices=[
            "inventory_optimization_crash_course",
            "inventory_optimization_cuopt_gpu",
            "inventory_optimization_large_benchmark",
        ],
        default="inventory_optimization_crash_course",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    bundle_vars = {
        "catalog": args.catalog,
        "schema": args.schema,
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "endpoint_name": args.endpoint_name,
        "gpu_model_name": args.gpu_model_name,
        "gpu_endpoint_name": args.gpu_endpoint_name,
        "gpu_serving_workload_type": args.gpu_serving_workload_type,
        "gpu_hardware_accelerator": args.gpu_hardware_accelerator,
        "gpu_environment_version": args.gpu_environment_version,
        "scenario_count": args.scenario_count,
        "small_sku_counts": args.small_sku_counts,
        "seed": args.seed,
        "deploy_endpoint": args.deploy_endpoint,
        "run_large_benchmark": args.run_large_benchmark,
        "large_product_count": args.large_product_count,
        "large_source_count": args.large_source_count,
        "large_dc_count": args.large_dc_count,
        "large_store_count": args.large_store_count,
        "large_sources_per_dc": args.large_sources_per_dc,
        "large_dcs_per_store": args.large_dcs_per_store,
        "large_time_limit_s": args.large_time_limit_s,
        "large_experiment_name": args.large_experiment_name,
        "large_benchmark_id": args.large_benchmark_id,
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
            resource_key=args.resource_key,
            variables=bundle_vars,
        )
    )


if __name__ == "__main__":
    main()
