from __future__ import annotations

import os
from typing import Any

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    AutoCaptureConfigInput,
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
    TrafficConfig,
)
from mlflow import MlflowClient


def resolve_registered_model_version(
    registered_model_name: str,
    *,
    alias: str = "Champion",
    explicit_version: str | None = None,
) -> str:
    if explicit_version:
        return str(explicit_version)

    client = MlflowClient()
    try:
        aliased = client.get_model_version_by_alias(registered_model_name, alias)
        return str(aliased.version)
    except Exception:
        pass

    versions = list(client.search_model_versions(f"name='{registered_model_name}'"))
    if not versions:
        raise ValueError(f"No versions found for registered model {registered_model_name}")
    latest = max(versions, key=lambda version_info: int(version_info.version))
    return str(latest.version)


def deploy_serving_endpoint(
    *,
    profile: str,
    endpoint_name: str,
    registered_model_name: str,
    model_version: str | None = None,
    alias: str = "Champion",
    workload_size: str = "Small",
    scale_to_zero_enabled: bool = True,
    auto_capture_catalog: str | None = None,
    auto_capture_schema: str | None = None,
    auto_capture_table_prefix: str = "or_crash_course",
) -> dict[str, Any]:
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    resolved_version = resolve_registered_model_version(
        registered_model_name,
        alias=alias,
        explicit_version=model_version,
    )
    served_model_name = f"{registered_model_name.split('.')[-1]}-{resolved_version}"

    served_entities = [
        ServedEntityInput(
            entity_name=registered_model_name,
            entity_version=resolved_version,
            name=served_model_name,
            workload_size=workload_size,
            scale_to_zero_enabled=scale_to_zero_enabled,
        )
    ]
    traffic_config = TrafficConfig(routes=[Route(served_model_name=served_model_name, traffic_percentage=100)])
    auto_capture = None
    if auto_capture_catalog and auto_capture_schema:
        auto_capture = AutoCaptureConfigInput(
            catalog_name=auto_capture_catalog,
            schema_name=auto_capture_schema,
            table_name_prefix=auto_capture_table_prefix,
            enabled=True,
        )

    workspace = WorkspaceClient(profile=profile)
    try:
        workspace.serving_endpoints.get(endpoint_name)
        workspace.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=served_entities,
            traffic_config=traffic_config,
            auto_capture_config=auto_capture,
        )
        action = "updated"
    except NotFound:
        workspace.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=served_entities,
                traffic_config=traffic_config,
                auto_capture_config=auto_capture,
            ),
        )
        action = "created"

    return {
        "action": action,
        "endpoint_name": endpoint_name,
        "registered_model_name": registered_model_name,
        "model_version": resolved_version,
        "served_model_name": served_model_name,
    }
