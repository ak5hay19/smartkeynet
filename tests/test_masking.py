"""Behavioral tests for `env.masking` (PLAN.md §4 "ACTION MASKING
(structural, inviolable)"; Hard Rule 2 -- floors are enforced by
masking, never by reward penalties, and threat signals may only raise
floors, never lower them).
"""

from __future__ import annotations

import itertools

from env.contracts import Action, N_ACTIONS, Request, SensitivityClass, ThreatPosture
from env.masking import PolicyTable, compute_mask, load_key_lifetime_config

MAX_KEY_AGE = load_key_lifetime_config()["max_key_age_steps"]


def make_request(sensitivity_class: int = 0, hybrid_mandatory: bool = False) -> Request:
    return Request(
        request_id="r0",
        step=0,
        tenant="t",
        service="svc",
        sensitivity_class=sensitivity_class,
        pqc_capable=True,
        hybrid_mandatory=hybrid_mandatory,
    )


# ---------------------------------------------------------------------------
# compute_mask
# ---------------------------------------------------------------------------


def test_actions_below_floor_are_masked():
    request = make_request()
    mask = compute_mask(request, floor=Action.SERVE_HYBRID, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=True)

    assert mask[Action.SERVE_CLASSICAL] == False  # noqa: E712
    assert mask[Action.SERVE_PQC] == False  # noqa: E712
    assert mask[Action.SERVE_HYBRID] == True  # noqa: E712


def test_reuse_masked_when_key_age_at_or_over_cap():
    request = make_request()

    over_cap = compute_mask(request, floor=Action.SERVE_CLASSICAL, key_age=MAX_KEY_AGE, max_key_age=MAX_KEY_AGE, pool_can_draw=True)
    assert over_cap[Action.REUSE] == False  # noqa: E712

    under_cap = compute_mask(request, floor=Action.SERVE_CLASSICAL, key_age=MAX_KEY_AGE - 1, max_key_age=MAX_KEY_AGE, pool_can_draw=True)
    assert under_cap[Action.REUSE] == True  # noqa: E712


def test_serve_hybrid_masked_when_pool_cannot_draw():
    request = make_request()

    cannot_draw = compute_mask(request, floor=Action.SERVE_CLASSICAL, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=False)
    assert cannot_draw[Action.SERVE_HYBRID] == False  # noqa: E712

    can_draw = compute_mask(request, floor=Action.SERVE_CLASSICAL, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=True)
    assert can_draw[Action.SERVE_HYBRID] == True  # noqa: E712


def test_nothing_masked_at_lowest_floor_pool_ok_key_fresh():
    request = make_request()
    mask = compute_mask(request, floor=Action.SERVE_CLASSICAL, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=True)

    assert mask.all()


def test_at_least_one_action_always_legal():
    """The env must never deadlock -- REKEY_NOW's action index (4) is
    always >= any tier floor (max index 2) and is never touched by the
    pool-draw or key-age rules, so it's always the escape hatch."""
    request = make_request()
    floors = [Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID]
    key_ages = [0.0, MAX_KEY_AGE / 2, MAX_KEY_AGE, MAX_KEY_AGE + 100]
    pool_states = [True, False]

    for floor, key_age, pool_can_draw in itertools.product(floors, key_ages, pool_states):
        mask = compute_mask(request, floor=floor, key_age=key_age, max_key_age=MAX_KEY_AGE, pool_can_draw=pool_can_draw)
        assert mask.any(), f"deadlock at floor={floor}, key_age={key_age}, pool_can_draw={pool_can_draw}"
        assert mask.shape == (N_ACTIONS,)


def test_serve_hybrid_masked_by_pool_even_when_it_is_the_floor():
    """Hard Rule 9: pool exhaustion never causes a silent downgrade --
    if floor is SERVE_HYBRID and the pool can't cover it, SERVE_HYBRID
    is masked (routing to the deferral queue), not swapped for a lower
    tier."""
    request = make_request(sensitivity_class=3, hybrid_mandatory=True)
    mask = compute_mask(request, floor=Action.SERVE_HYBRID, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=False)

    assert mask[Action.SERVE_HYBRID] == False  # noqa: E712
    assert mask[Action.SERVE_CLASSICAL] == False  # noqa: E712
    assert mask[Action.SERVE_PQC] == False  # noqa: E712


# ---------------------------------------------------------------------------
# PolicyTable.floor monotonicity
# ---------------------------------------------------------------------------


def test_floor_monotonic_in_sensitivity_class():
    table = PolicyTable()
    for posture in ThreatPosture:
        floors = [int(table.floor(sc, posture)) for sc in SensitivityClass]
        assert floors == sorted(floors), f"non-monotonic in class at posture={posture}: {floors}"


def test_floor_monotonic_in_threat_posture():
    table = PolicyTable()
    for sc in SensitivityClass:
        floors = [int(table.floor(sc, posture)) for posture in ThreatPosture]
        assert floors == sorted(floors), f"non-monotonic in posture at class={sc}: {floors}"


def test_s3_never_floors_below_pqc():
    table = PolicyTable()
    for posture in ThreatPosture:
        assert int(table.floor(SensitivityClass.S3, posture)) >= int(Action.SERVE_PQC)


def test_s0_can_floor_at_classical_when_calm():
    table = PolicyTable()
    assert table.floor(SensitivityClass.S0, ThreatPosture.CALM) == Action.SERVE_CLASSICAL


# ---------------------------------------------------------------------------
# PolicyTable.ratchet_up
# ---------------------------------------------------------------------------


def test_ratchet_up_only_raises_never_lowers_subsequent_floor():
    table = PolicyTable()
    before = table.floor(SensitivityClass.S0, ThreatPosture.CALM)

    table.ratchet_up(ThreatPosture.HIGH)
    after = table.floor(SensitivityClass.S0, ThreatPosture.CALM)

    assert int(after) >= int(before)


def test_ratchet_up_sticks_even_if_later_posture_argument_is_lower():
    table = PolicyTable()
    table.ratchet_up(ThreatPosture.HIGH)

    floor_with_calm_arg = table.floor(SensitivityClass.S1, ThreatPosture.CALM)
    floor_at_high_directly = table.floor(SensitivityClass.S1, ThreatPosture.HIGH)

    assert floor_with_calm_arg == floor_at_high_directly


def test_ratchet_up_is_a_noop_when_not_higher_than_current():
    table = PolicyTable()
    table.ratchet_up(ThreatPosture.HIGH)
    floor_after_high = table.floor(SensitivityClass.S2, ThreatPosture.CALM)

    table.ratchet_up(ThreatPosture.CALM)  # lower than current ratchet -- must not change anything
    floor_after_noop = table.floor(SensitivityClass.S2, ThreatPosture.CALM)

    assert floor_after_high == floor_after_noop


def test_ratchet_up_never_lowers_floor_across_all_class_posture_pairs():
    table = PolicyTable()
    before = {
        (sc, posture): int(table.floor(sc, posture))
        for sc in SensitivityClass
        for posture in ThreatPosture
    }

    table.ratchet_up(ThreatPosture.ELEVATED)

    for (sc, posture), before_floor in before.items():
        after_floor = int(table.floor(sc, posture))
        assert after_floor >= before_floor
