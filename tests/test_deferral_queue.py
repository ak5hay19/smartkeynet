"""Behavioral tests for `env.deferral_queue` (PLAN.md Addition C,
implementing Hard Rule 9: pool exhaustion never causes a downgrade --
it causes a deferral, logged as a regret event).
"""

from __future__ import annotations

from env.contracts import Action, Request
from env.deferral_queue import DeferralQueue


def make_request(
    request_id: str,
    sensitivity_class: int,
    tenant: str = "hospital",
    step: int = 0,
) -> Request:
    return Request(
        request_id=request_id,
        step=step,
        tenant=tenant,
        service="svc",
        sensitivity_class=sensitivity_class,
        pqc_capable=True,
        hybrid_mandatory=True,
    )


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


def test_higher_sensitivity_class_served_before_lower():
    queue = DeferralQueue()
    queue.enqueue(
        make_request("low", sensitivity_class=1, step=0),
        bits_required=100.0,
        step=0,
        pool_fill_at_onset=0.0,
    )
    queue.enqueue(
        make_request("high", sensitivity_class=3, step=1),
        bits_required=100.0,
        step=1,
        pool_fill_at_onset=0.0,
    )

    # pool can only cover one request's worth of bits at a time
    servable = queue.pop_servable(lambda bits: bits <= 100.0)

    assert len(servable) == 1
    assert servable[0].request["request_id"] == "high"
    assert len(queue) == 1  # "low" still queued


def test_fifo_ordering_within_same_class():
    queue = DeferralQueue()
    for i, rid in enumerate(["r0", "r1", "r2"]):
        queue.enqueue(
            make_request(rid, sensitivity_class=2, step=i),
            bits_required=50.0,
            step=i,
            pool_fill_at_onset=0.0,
        )

    # pool covers exactly two of the three requests
    servable = queue.pop_servable(lambda bits: bits <= 100.0)

    assert [q.request["request_id"] for q in servable] == ["r0", "r1"]
    assert len(queue) == 1
    remaining = queue.pop_servable(lambda bits: True)
    assert remaining[0].request["request_id"] == "r2"


def test_pop_servable_respects_cumulative_pool_headroom():
    """Two same-priority requests each fit individually but not together
    -- pop_servable must not double-commit the pool within one pass."""
    queue = DeferralQueue()
    queue.enqueue(
        make_request("a", sensitivity_class=1, step=0),
        bits_required=80.0,
        step=0,
        pool_fill_at_onset=0.0,
    )
    queue.enqueue(
        make_request("b", sensitivity_class=1, step=1),
        bits_required=80.0,
        step=1,
        pool_fill_at_onset=0.0,
    )

    servable = queue.pop_servable(lambda bits: bits <= 100.0)

    assert len(servable) == 1
    assert servable[0].request["request_id"] == "a"
    assert len(queue) == 1


def test_pop_servable_skips_unfit_head_of_line_for_smaller_lower_priority():
    """A high-priority request that doesn't fit shouldn't block a smaller
    lower-priority request that does."""
    queue = DeferralQueue()
    queue.enqueue(
        make_request("big-high", sensitivity_class=3, step=0),
        bits_required=1000.0,
        step=0,
        pool_fill_at_onset=0.0,
    )
    queue.enqueue(
        make_request("small-low", sensitivity_class=1, step=1),
        bits_required=10.0,
        step=1,
        pool_fill_at_onset=0.0,
    )

    servable = queue.pop_servable(lambda bits: bits <= 10.0)

    assert [q.request["request_id"] for q in servable] == ["small-low"]
    assert len(queue) == 1


# ---------------------------------------------------------------------------
# Regret event onset semantics
# ---------------------------------------------------------------------------


def test_regret_event_fires_once_on_enqueue_not_on_every_tick():
    queue = DeferralQueue()
    event = queue.enqueue(
        make_request("r0", sensitivity_class=2, step=5),
        bits_required=100.0,
        step=5,
        pool_fill_at_onset=42.0,
    )

    assert event["step"] == 5
    assert event["request_id"] == "r0"
    assert event["tenant"] == "hospital"
    assert event["sensitivity_class"] == 2
    assert event["policy_floor"] == int(Action.SERVE_HYBRID)
    assert event["pool_fill_at_onset"] == 42.0

    # ticking repeatedly must never produce another regret event -- tick()
    # only returns DeferredCriticalStep entries.
    for step in range(6, 9):
        deferred_steps = queue.tick(step)
        assert len(deferred_steps) == 1
        assert "policy_floor" not in deferred_steps[0]
        assert deferred_steps[0]["request_id"] == "r0"

    assert queue.tick(9)[0]["steps_waited"] == 4


def test_tick_returns_one_entry_per_still_queued_request():
    queue = DeferralQueue()
    queue.enqueue(
        make_request("r0", sensitivity_class=1, step=0),
        bits_required=10.0,
        step=0,
        pool_fill_at_onset=0.0,
    )
    queue.enqueue(
        make_request("r1", sensitivity_class=2, step=0),
        bits_required=10.0,
        step=0,
        pool_fill_at_onset=0.0,
    )

    entries = queue.tick(1)
    assert {e["request_id"] for e in entries} == {"r0", "r1"}
    assert all(e["steps_waited"] == 1 for e in entries)


# ---------------------------------------------------------------------------
# Serving once covered
# ---------------------------------------------------------------------------


def test_deferred_request_served_once_pool_can_cover_it():
    queue = DeferralQueue()
    queue.enqueue(
        make_request("r0", sensitivity_class=3, step=0),
        bits_required=500.0,
        step=0,
        pool_fill_at_onset=0.0,
    )

    assert queue.pop_servable(lambda bits: bits <= 100.0) == []
    assert len(queue) == 1

    served = queue.pop_servable(lambda bits: bits <= 500.0)
    assert len(served) == 1
    assert served[0].request["request_id"] == "r0"
    assert len(queue) == 0


# ---------------------------------------------------------------------------
# Hard Rule 9 -- never downgraded while queued
# ---------------------------------------------------------------------------


def test_sensitivity_class_and_floor_never_change_while_queued():
    request = make_request("r0", sensitivity_class=3, step=0)
    queue = DeferralQueue()
    event = queue.enqueue(request, bits_required=200.0, step=0, pool_fill_at_onset=0.0)
    assert event["sensitivity_class"] == 3
    assert event["policy_floor"] == int(Action.SERVE_HYBRID)

    for step in range(1, 5):
        queue.tick(step)

    served = queue.pop_servable(lambda bits: bits <= 200.0)
    assert served[0].request["sensitivity_class"] == 3
    assert served[0].request["hybrid_mandatory"] is True
    # never silently served for fewer bits than required
    assert served[0].bits_required == 200.0


# ---------------------------------------------------------------------------
# §S2 -- SLA breach never downgrades; max_wait_steps
# ---------------------------------------------------------------------------


def test_sla_breach_does_not_downgrade():
    """§S2: "if r.wait_steps > sla_max_steps: emit sla_breach  # still NEVER
    downgraded".

    Hard Rule 9 trades a downgrade for a delay. An SLA breach is the size of
    that trade -- a reportable availability failure -- and explicitly not a
    licence to serve weaker key material. The assertion below is that breaching
    changes nothing about the request's floor or its presence in the queue.
    """
    queue = DeferralQueue()
    request = make_request(request_id="slow", sensitivity_class=3)
    queue.enqueue(request, bits_required=256.0, step=0, pool_fill_at_onset=0.0)

    for step in range(1, 51):
        queue.tick(step)

    breaches = queue.sla_breaches(sla_max_steps=20)
    assert len(breaches) == 1
    assert queue.max_wait_steps == 50
    # The breached request is still queued, still hybrid-mandatory, still at
    # its original class -- nothing about the breach weakened it.
    assert len(queue) == 1
    assert breaches[0].request["sensitivity_class"] == 3
    assert breaches[0].bits_required == 256.0


def test_no_sla_breach_before_the_threshold():
    queue = DeferralQueue()
    queue.enqueue(
        make_request(request_id="x", sensitivity_class=3),
        bits_required=256.0,
        step=0,
        pool_fill_at_onset=0.0,
    )
    for step in range(1, 6):
        queue.tick(step)
    assert queue.sla_breaches(sla_max_steps=20) == []
    assert queue.max_wait_steps == 5


def test_max_wait_steps_is_zero_on_an_empty_queue():
    assert DeferralQueue().max_wait_steps == 0
