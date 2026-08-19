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
    max_wait_steps: int = 0
    """Longest any single deferred request waited. A mean wait hides the
    tail, and the tail is what an SLA is written against (§S2)."""
    sla_breaches: int = 0
    """Deferrals that waited past `sla_max_steps`. Reported because Hard Rule 9
    trades a downgrade for a delay, and this is the size of that trade."""
    attributed_fraction: float = 0.0
    """`attributed_keys / regret_keys` -- how much of the episode's regret is
    traceable to the agent's own discretionary spending, as opposed to the
    environment simply not having enough key material (§S2)."""


def compute_episode_metrics(
    regret_events: list[RegretEvent],
    deferred_steps: list[DeferredCriticalStep],
    forced_rekeys: list[ForcedRekey],
    total_rekeys: int,
    total_requests: int,
    discretionary_hybrid_serves: int,
    max_wait_steps: int = 0,
    sla_breaches: int = 0,
    attributed_fraction: float = 0.0,
) -> EpisodeMetrics:
    """Aggregate one episode's event log into `EpisodeMetrics`.

    `regret_events` is counted once per deferral onset (one entry per
    `DeferralQueue.enqueue()` call); `deferred_steps` is counted once
    per still-waiting tick (one entry per `DeferralQueue.tick()` call
    per queued request) -- these are deliberately different counters
    even though they're both driven by the same underlying deferrals.
    """
    rekeys_per_100_requests = (total_rekeys / total_requests) * 100.0 if total_requests > 0 else 0.0
    forced_rekey_ratio = len(forced_rekeys) / total_rekeys if total_rekeys > 0 else 0.0

    return EpisodeMetrics(
        regret_events=len(regret_events),
        deferred_critical_steps=len(deferred_steps),
        rekeys_per_100_requests=rekeys_per_100_requests,
        forced_rekey_ratio=forced_rekey_ratio,
        max_wait_steps=max_wait_steps,
        sla_breaches=sla_breaches,
        attributed_fraction=attributed_fraction,
        discretionary_hybrid_serves=discretionary_hybrid_serves,
    )


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

    `hybrid_serve_log` entries are not a frozen `contracts.py` type
    (this log is analysis/plots only -- PLAN.md Addition C -- and
    never feeds reward or state). Each entry is expected to carry at
    least: `{"step": int, "bits": float, "discretionary": bool}`,
    where `discretionary` marks a hybrid serve that wasn't itself
    hybrid-mandatory (i.e. one the agent chose to spend pool bits on,
    as opposed to one it was required to serve).

    Attribution procedure: process regret events in step order. For
    each, attribute every *not-yet-claimed* discretionary serve that
    happened strictly before that event's step -- each discretionary
    serve's bits are claimed by at most one regret event, ever. This
    is what guarantees the unit test invariant (Addition C: "bits
    attributed <= bits spent"): summed across every `AttributionEntry`
    returned, `bits_attributed` can never exceed the total bits spent
    across `hybrid_serve_log`'s discretionary entries, because no
    entry's bits are ever attributed twice.

    `RegretEvent` doesn't carry the deferred request's `bits_required`
    (that lives on `env/deferral_queue.py`'s `QueuedRequest`, not the
    event log), so this can't reconstruct an exact "these specific
    bits would have covered it" trace -- it reports every unclaimed
    discretionary spend that preceded the deferral, which is the
    honest amount of information available from the event log alone.
    """
    discretionary = [entry for entry in hybrid_serve_log if entry.get("discretionary", False)]
    claimed: set[int] = set()

    entries: list[AttributionEntry] = []
    for event in sorted(regret_events, key=lambda e: e["step"]):
        attributed_steps: list[int] = []
        bits_attributed = 0.0
        for idx, serve in enumerate(discretionary):
            if idx in claimed:
                continue
            if serve["step"] < event["step"]:
                claimed.add(idx)
                attributed_steps.append(serve["step"])
                bits_attributed += serve["bits"]

        entries.append(
            AttributionEntry(
                regret_event=event,
                attributed_serve_steps=attributed_steps,
                bits_attributed=bits_attributed,
            )
        )

    return entries
