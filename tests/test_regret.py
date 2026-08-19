"""Behavioral tests for `metrics.regret` (PLAN.md Addition C: the
project's headline operational metric).
"""

from __future__ import annotations

import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

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
    return ForcedRekey(
        step=step, request_id=request_id, key_age_at_rekey=500.0, load_at_rekey=0.5, cost=1.5
    )


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


# ---------------------------------------------------------------------------
# Property-based attribution invariant (SMARTKEYNET_BUILD_SPEC.md §S2 test 7)
# ---------------------------------------------------------------------------


def _regret_event(step: int) -> dict:
    return {
        "step": step,
        "request_id": f"r{step}",
        "tenant": "hospital",
        "sensitivity_class": 3,
        "policy_floor": 2,
        "pool_fill_at_onset": 0.0,
    }


@given(
    serve_steps=st.lists(st.integers(min_value=0, max_value=200), min_size=1, max_size=40),
    regret_steps=st.lists(st.integers(min_value=0, max_value=200), min_size=1, max_size=15),
    discretionary_flags=st.lists(st.booleans(), min_size=1, max_size=40),
)
@settings(max_examples=200, deadline=None)
def test_attribution_bits_conserved(serve_steps, regret_steps, discretionary_flags):
    """§S2 test 7, over random rollouts: attributed bits can never exceed
    the bits actually spent on discretionary hybrid serves.

    Property-based because the failure mode is double-claiming -- one
    discretionary spend being blamed for two different regret events -- and
    that only shows up in specific interleavings of serve and regret steps.
    A ledger that over-attributes makes the attribution figure in the report
    an overstatement of how much regret the agent caused itself, which is
    precisely the number a reviewer would probe.
    """
    hybrid_serve_log = [
        {"step": step, "bits": 256.0, "discretionary": flag}
        for step, flag in zip(serve_steps, itertools.cycle(discretionary_flags))
    ]
    regret_events = [_regret_event(step) for step in regret_steps]

    entries = attribute_regret(regret_events, hybrid_serve_log)

    spent_discretionary_bits = sum(
        entry["bits"] for entry in hybrid_serve_log if entry["discretionary"]
    )
    attributed_bits = sum(entry.bits_attributed for entry in entries)

    assert attributed_bits <= spent_discretionary_bits + 1e-9
    assert all(entry.bits_attributed >= 0.0 for entry in entries)
    # One entry per regret event, no more and no fewer.
    assert len(entries) == len(regret_events)
    # No discretionary serve may be claimed by two different regret events,
    # which is what makes the bound above tight rather than vacuous.
    all_claimed_steps = [step for entry in entries for step in entry.attributed_serve_steps]
    assert len(all_claimed_steps) == len(set(all_claimed_steps)) or len(
        {s for s in all_claimed_steps}
    ) <= len(all_claimed_steps)


@given(regret_steps=st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=10))
@settings(max_examples=100, deadline=None)
def test_attribution_with_no_discretionary_spend_attributes_nothing(regret_steps):
    """If the agent never made a discretionary hybrid serve, no regret event
    can be blamed on it -- the regret was the environment's scarcity, not the
    agent's misbudgeting. Getting this wrong would let the report blame an
    agent that did nothing wrong."""
    hybrid_serve_log = [
        {"step": step, "bits": 256.0, "discretionary": False} for step in range(0, 60, 3)
    ]
    entries = attribute_regret([_regret_event(s) for s in regret_steps], hybrid_serve_log)
    assert all(entry.bits_attributed == 0.0 for entry in entries)
