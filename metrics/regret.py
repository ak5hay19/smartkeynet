"""
metrics/regret.py

Per-episode regret & churn metrics (PLAN.md Addition C). Owned by
Person B (split.md §1).

Computed from the event log emitted by `env/deferral_queue.py` and
`env/environment.py` (`RegretEvent`, `DeferredCriticalStep`,
`ForcedRekey` -- see `env/contracts.py`). This is the project's
headline operational metric (PLAN.md §6 Demo Beat 2, §9 closing table).
"""

from __future__ import annotations

from dataclasses import dataclass

from env.contracts import DeferredCriticalStep, ForcedRekey, RegretEvent


@dataclass
class EpisodeMetrics:
    """Per-episode summary (PLAN.md Addition C "Metrics module")."""

    regret_events: int
    deferred_critical_steps: int
    rekeys_per_100_requests: float
    forced_rekey_ratio: float  # forced / total rekeys
    discretionary_hybrid_serves: int


def compute_episode_metrics(
    regret_events: list[RegretEvent],
    deferred_steps: list[DeferredCriticalStep],
    forced_rekeys: list[ForcedRekey],
    total_rekeys: int,
    total_requests: int,
    discretionary_hybrid_serves: int,
) -> EpisodeMetrics:
    """Aggregate one episode's event log into `EpisodeMetrics`."""
    raise NotImplementedError


@dataclass
class AttributionEntry:
    """One row of the retrospective attribution log (PLAN.md Addition
    C): for a regret event, which earlier discretionary hybrid serves
    consumed the pool bits that would have covered it.

    Analysis/plots only -- never enters reward or state.
    """

    regret_event: RegretEvent
    attributed_serve_steps: list[int]
    bits_attributed: float


def attribute_regret(
    regret_events: list[RegretEvent], hybrid_serve_log: list[dict]
) -> list[AttributionEntry]:
    """Build the retrospective attribution log.

    Unit test invariant (Addition C): bits attributed <= bits spent.
    """
    raise NotImplementedError
