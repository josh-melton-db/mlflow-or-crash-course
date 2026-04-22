from __future__ import annotations

from typing import Iterable

import numpy as np

from .types import PortfolioScenario


def example_scenario() -> PortfolioScenario:
    return PortfolioScenario.from_lists(
        scenario_id="example_capital_plan",
        project_ids=["warehouse", "pricing-engine", "route-optimizer", "mobile-app", "etl-refresh"],
        values=[130, 90, 160, 75, 55],
        costs=[40, 30, 45, 25, 20],
        hours=[20, 18, 22, 15, 12],
        budget=95,
        capacity=48,
    )


def generate_benchmark_scenarios(seed: int = 7, scenario_count: int = 6) -> list[PortfolioScenario]:
    rng = np.random.default_rng(seed)
    item_counts = np.linspace(18, 90, num=scenario_count, dtype=int)
    scenarios: list[PortfolioScenario] = []

    for index, item_count in enumerate(item_counts, start=1):
        costs = rng.integers(12, 70, size=item_count)
        hours = rng.integers(6, 32, size=item_count)

        # Blend multiple signals so the benchmark has a few non-obvious trade-offs.
        values = (
            rng.integers(45, 220, size=item_count)
            + (costs * rng.integers(1, 4, size=item_count))
            + (hours * rng.integers(2, 6, size=item_count))
        )

        budget = int(costs.sum() * rng.uniform(0.42, 0.56))
        capacity = int(hours.sum() * rng.uniform(0.38, 0.52))

        scenarios.append(
            PortfolioScenario.from_lists(
                scenario_id=f"scenario_{index:02d}_{item_count}projects",
                project_ids=[f"P{project_id + 1:03d}" for project_id in range(item_count)],
                values=values.tolist(),
                costs=costs.tolist(),
                hours=hours.tolist(),
                budget=budget,
                capacity=capacity,
            )
        )

    return scenarios


def build_request_rows(scenarios: Iterable[PortfolioScenario]) -> list[dict[str, object]]:
    return [scenario.to_request_record() for scenario in scenarios]
