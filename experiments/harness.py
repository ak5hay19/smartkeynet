"""
experiments/harness.py

Comparison harness: runs any `Policy` (see `agents/baselines.py`'s
`Policy` protocol) across scenarios S1-S6 and records
`metrics/regret.py` output (PLAN.md §5 scenario grid; §10 kickoff step
6). Owned by Person C (split.md §1).

Build this *before* tuning the DQN (Hard Rule 7) -- baselines need to
be comparable from day one, not bolted on after the agent looks good.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.baselines import Policy
from metrics.regret import EpisodeMetrics


@dataclass
class ScenarioResult:
    """One (policy, scenario, seed) run's outcome."""

    scenario: str
    seed: int
    episode_metrics: EpisodeMetrics
    p99_latency: float
    pool_exhaustion_events: int
    floor_violations: int  # must be 0 for any masked policy, by construction


def run_scenario(
    policy: Policy, scenario: str, config: dict[str, Any], seed: int
) -> ScenarioResult:
    """Run one episode of `scenario` with `policy`, return its
    `ScenarioResult`."""
    raise NotImplementedError


def run_grid(
    policies: dict[str, Policy],
    scenarios: list[str],
    config: dict[str, Any],
    seeds: list[int],
) -> list[ScenarioResult]:
    """Run every (policy, scenario, seed) combination (PLAN.md §6
    closing table: Agent vs. always-PQC vs. always-hybrid vs.
    static-threshold vs. random, across S1-S4 + S6)."""
    raise NotImplementedError
