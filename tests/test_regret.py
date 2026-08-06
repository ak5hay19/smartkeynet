"""Behavioral tests for `metrics.regret` (PLAN.md Addition C: the
project's headline operational metric).
"""

from __future__ import annotations

from env.contracts import DeferredCriticalStep, ForcedRekey, RegretEvent
from metrics.regret import attribute_regret, compute_episode_metrics


def make_regret_event(step: int, request_id: str, sensitivity_class: int = 2) -> RegretEvent:
    return RegretEvent(
        step=step,
        request_id=request_id,
        tenant="hospital",
        sensitivity_class=sensitivity_class,
        policy_floor=2,
        pool_fill_at_onset=0.0,
    )


def make_deferred_step(step: int, request_id: str, steps_waited: int) -> DeferredCriticalStep:
    return DeferredCriticalStep(step=step, request_id=request_id, steps_waited=steps_waited)


def make_forced_rekey(step: int, request_id: str) -> ForcedRekey:
    return ForcedRekey(step=step, request_id=request_id, key_age_at_rekey=500.0, load_at_rekey=0.5, cost=1.5)


# ---------------------------------------------------------------------------
# compute_episode_metrics
# ---------------------------------------------------------------------------


def test_regret_events_count_once_per_onset_not_per_waiting_step():
    regret_events = [make_regret_event(0, "r0"), make_regret_event(3, "r1")]
    deferred_steps = [
        make_deferred_step(1, "r0", 1),
        make_deferred_step(2, "r0", 2),
        make_deferred_step(3, "r0", 3),
        make_deferred_step(4, "r1", 1),
        make_deferred_step(5, "r1", 2),
    ]

    metrics = compute_episode_metrics(
        regret_events=regret_events,
        deferred_steps=deferred_steps,
        forced_rekeys=[],
        total_rekeys=0,
        total_requests=10,
        discretionary_hybrid_serves=0,
    )

    assert metrics.regret_events == 2  # onsets only, not one per waiting step


def test_deferred_critical_steps_increments_every_waiting_step():
    deferred_steps = [make_deferred_step(s, "r0", s) for s in range(1, 6)]

    metrics = compute_episode_metrics(
        regret_events=[],
        deferred_steps=deferred_steps,
        forced_rekeys=[],
        total_rekeys=0,
        total_requests=10,
        discretionary_hybrid_serves=0,
    )

    assert metrics.deferred_critical_steps == 5


def test_forced_rekey_ratio_computes_correctly():
    forced = [make_forced_rekey(s, f"r{s}") for s in range(3)]

    metrics = compute_episode_metrics(
        regret_events=[],
        deferred_steps=[],
        forced_rekeys=forced,
        total_rekeys=10,
        total_requests=100,
        discretionary_hybrid_serves=0,
    )

    assert metrics.forced_rekey_ratio == 0.3


def test_forced_rekey_ratio_zero_when_no_rekeys():
    metrics = compute_episode_metrics(
        regret_events=[],
        deferred_steps=[],
        forced_rekeys=[],
        total_rekeys=0,
        total_requests=100,
        discretionary_hybrid_serves=0,
    )

    assert metrics.forced_rekey_ratio == 0.0


def test_rekeys_per_100_requests_computes_correctly():
    metrics = compute_episode_metrics(
        regret_events=[],
        deferred_steps=[],
        forced_rekeys=[],
        total_rekeys=25,
        total_requests=500,
        discretionary_hybrid_serves=0,
    )

    assert metrics.rekeys_per_100_requests == 5.0


def test_rekeys_per_100_requests_zero_when_no_requests():
    metrics = compute_episode_metrics(
        regret_events=[],
        deferred_steps=[],
        forced_rekeys=[],
        total_rekeys=0,
        total_requests=0,
        discretionary_hybrid_serves=0,
    )

    assert metrics.rekeys_per_100_requests == 0.0


def test_discretionary_hybrid_serves_passes_through():
    metrics = compute_episode_metrics(
        regret_events=[],
        deferred_steps=[],
        forced_rekeys=[],
        total_rekeys=0,
        total_requests=0,
        discretionary_hybrid_serves=17,
    )

    assert metrics.discretionary_hybrid_serves == 17


# ---------------------------------------------------------------------------
# attribute_regret
# ---------------------------------------------------------------------------


def test_attribution_bits_never_exceed_bits_spent():
    hybrid_serve_log = [
        {"step": 1, "bits": 100.0, "discretionary": True},
        {"step": 2, "bits": 50.0, "discretionary": True},
        {"step": 3, "bits": 200.0, "discretionary": False},  # not discretionary
        {"step": 4, "bits": 75.0, "discretionary": True},
    ]
    regret_events = [make_regret_event(5, "r0")]

    entries = attribute_regret(regret_events, hybrid_serve_log)

    total_spent_discretionary = sum(e["bits"] for e in hybrid_serve_log if e["discretionary"])
    total_attributed = sum(entry.bits_attributed for entry in entries)
    assert total_attributed <= total_spent_discretionary


def test_attribution_only_considers_discretionary_serves():
    hybrid_serve_log = [
        {"step": 1, "bits": 100.0, "discretionary": False},
    ]
    regret_events = [make_regret_event(5, "r0")]

    entries = attribute_regret(regret_events, hybrid_serve_log)

    assert entries[0].bits_attributed == 0.0
    assert entries[0].attributed_serve_steps == []


def test_attribution_only_considers_serves_before_the_event():
    hybrid_serve_log = [
        {"step": 10, "bits": 100.0, "discretionary": True},  # after the regret event
    ]
    regret_events = [make_regret_event(5, "r0")]

    entries = attribute_regret(regret_events, hybrid_serve_log)

    assert entries[0].bits_attributed == 0.0


def test_attribution_does_not_double_count_a_serve_across_events():
    hybrid_serve_log = [
        {"step": 1, "bits": 100.0, "discretionary": True},
    ]
    regret_events = [make_regret_event(2, "r0"), make_regret_event(5, "r1")]

    entries = attribute_regret(regret_events, hybrid_serve_log)

    total_attributed = sum(entry.bits_attributed for entry in entries)
    assert total_attributed == 100.0  # claimed once, by the earlier event only
    first_event_entry = next(e for e in entries if e.regret_event["request_id"] == "r0")
    second_event_entry = next(e for e in entries if e.regret_event["request_id"] == "r1")
    assert first_event_entry.bits_attributed == 100.0
    assert second_event_entry.bits_attributed == 0.0
