"""Behavioral tests for `env.masking` (PLAN.md §4 "ACTION MASKING
(structural, inviolable)"; Hard Rule 2 -- floors are enforced by
masking, never by reward penalties, and threat signals may only raise
floors, never lower them).
"""

from __future__ import annotations

import itertools

from env.contracts import Action, N_ACTIONS, Request, SensitivityClass, ThreatPosture
from env.masking import (
    PolicyTable,
    compute_mask,
    effective_floor_for,
    load_key_lifetime_config,
)

MAX_KEY_AGE = load_key_lifetime_config()["max_key_age_steps"]


def make_request(
    sensitivity_class: int = 0,
    hybrid_mandatory: bool = False,
    pqc_capable: bool = True,
) -> Request:
    return Request(
        request_id="r0",
        step=0,
        tenant="t",
        service="svc",
        sensitivity_class=sensitivity_class,
        pqc_capable=pqc_capable,
        hybrid_mandatory=hybrid_mandatory,
    )


# ---------------------------------------------------------------------------
# hybrid_mandatory raises the effective floor (2026-08-15)
# ---------------------------------------------------------------------------


def test_hybrid_mandatory_raises_the_effective_floor():
    """Regression test. Before 2026-08-15 `hybrid_mandatory` only
    triggered the env's deferral pre-screen and was invisible to
    masking, so whenever the pool could cover the request nothing
    forced a hybrid serve -- `always_pqc` served hybrid-mandatory
    requests at PQC with zero recorded violations."""
    request = make_request(sensitivity_class=3, hybrid_mandatory=True)
    mask = compute_mask(
        request, floor=Action.SERVE_CLASSICAL, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=True
    )

    assert not mask[Action.SERVE_CLASSICAL]
    assert not mask[Action.SERVE_PQC]
    assert mask[Action.SERVE_HYBRID]


def test_hybrid_mandatory_never_lowers_a_higher_floor():
    """`max`-only, so it cannot relax anything (Hard Rule 2)."""
    for floor in Action:
        plain = effective_floor_for(make_request(hybrid_mandatory=False), floor)
        mandatory = effective_floor_for(make_request(hybrid_mandatory=True), floor)
        assert int(mandatory) >= int(plain) == int(floor)


def test_non_mandatory_request_keeps_the_table_floor():
    request = make_request(hybrid_mandatory=False)
    assert effective_floor_for(request, Action.SERVE_PQC) == Action.SERVE_PQC


# ---------------------------------------------------------------------------
# pqc_capable interoperability masking (spec §S4 rule 2)
# ---------------------------------------------------------------------------


def test_legacy_endpoint_masks_pqc_and_hybrid():
    request = make_request(pqc_capable=False)
    mask = compute_mask(
        request, floor=Action.SERVE_CLASSICAL, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=True
    )

    assert mask[Action.SERVE_CLASSICAL]  # the only interoperable option stays legal
    assert not mask[Action.SERVE_PQC]
    assert not mask[Action.SERVE_HYBRID]


def test_legacy_endpoint_still_has_a_legal_action_at_every_posture():
    """Liveness under the exemption: without it, a legacy S0 flow at
    HIGH posture would have nothing legal at all (floor PQC, but PQC
    un-negotiable)."""
    table = PolicyTable()
    request = make_request(sensitivity_class=0, pqc_capable=False)
    for posture in ThreatPosture:
        floor = table.floor(SensitivityClass.S0, posture, pqc_capable=False)
        mask = compute_mask(
            request, floor=floor, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=True
        )
        assert mask.any(), f"legacy flow deadlocked at posture={posture}"


def test_legacy_exemption_cannot_be_triggered_by_any_threat_posture():
    """The exemption keys off a static capability fact, never off a
    threat signal -- so no posture, forecast or adversarial trace can
    reach it. This is what keeps it outside Hard Rule 2's scope."""
    table = PolicyTable()
    for posture in ThreatPosture:
        table.ratchet_up(posture)
        for sc in SensitivityClass:
            capable_floor = table.floor(sc, posture, pqc_capable=True)
            legacy_floor = table.floor(sc, posture, pqc_capable=False)
            # a PQC-capable flow's floor is never relaxed by anything
            assert int(capable_floor) >= int(PolicyTable().floor(sc, ThreatPosture.CALM))
            # the legacy floor is constant: it does not vary with posture at all
            assert legacy_floor == Action.SERVE_CLASSICAL


def test_pqc_capable_flow_is_unaffected_by_the_exemption():
    table = PolicyTable()
    for sc in SensitivityClass:
        for posture in ThreatPosture:
            assert table.floor(sc, posture, pqc_capable=True) == table.floor(sc, posture)


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

    over_cap = compute_mask(
        request, floor=Action.SERVE_CLASSICAL, key_age=MAX_KEY_AGE, max_key_age=MAX_KEY_AGE,
        pool_can_draw=True, active_key_tier=Action.SERVE_CLASSICAL,
    )
    assert over_cap[Action.REUSE] == False  # noqa: E712

    under_cap = compute_mask(
        request, floor=Action.SERVE_CLASSICAL, key_age=MAX_KEY_AGE - 1, max_key_age=MAX_KEY_AGE,
        pool_can_draw=True, active_key_tier=Action.SERVE_CLASSICAL,
    )
    assert under_cap[Action.REUSE] == True  # noqa: E712


# ---------------------------------------------------------------------------
# REUSE vs the floor (spec §S4 masking rule 4, implemented 2026-08-15)
# ---------------------------------------------------------------------------


def test_reuse_masked_when_there_is_no_active_key():
    request = make_request()
    mask = compute_mask(
        request, floor=Action.SERVE_CLASSICAL, key_age=0.0, max_key_age=MAX_KEY_AGE,
        pool_can_draw=True, active_key_tier=None,
    )
    assert not mask[Action.REUSE]


def test_reuse_masked_when_the_active_key_is_below_the_floor():
    """Regression test for a Hard Rule 2 hole. Reusing a classical key
    under a hybrid floor is a floor violation reached through the one
    action whose tier is state-dependent; measured on an S2 episode
    before the fix, 1,090 of 3,000 REUSE actions did exactly that while
    `floor_violations` reported zero."""
    request = make_request()
    for floor, key_tier, expected_legal in [
        (Action.SERVE_HYBRID, Action.SERVE_CLASSICAL, False),
        (Action.SERVE_HYBRID, Action.SERVE_PQC, False),
        (Action.SERVE_HYBRID, Action.SERVE_HYBRID, True),
        (Action.SERVE_PQC, Action.SERVE_CLASSICAL, False),
        (Action.SERVE_PQC, Action.SERVE_PQC, True),
        (Action.SERVE_PQC, Action.SERVE_HYBRID, True),
        (Action.SERVE_CLASSICAL, Action.SERVE_CLASSICAL, True),
    ]:
        mask = compute_mask(
            request, floor=floor, key_age=0.0, max_key_age=MAX_KEY_AGE,
            pool_can_draw=True, active_key_tier=key_tier,
        )
        assert bool(mask[Action.REUSE]) is expected_legal, (
            f"floor={floor.name}, key_tier={key_tier.name}"
        )


def test_reuse_of_a_high_tier_key_survives_a_floor_ratchet():
    """The anticipation payoff: a key provisioned at hybrid while the
    pool was healthy stays reusable after floors rise, so no pool draw
    is needed at the worst possible moment."""
    request = make_request()
    for floor in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID):
        mask = compute_mask(
            request, floor=floor, key_age=0.0, max_key_age=MAX_KEY_AGE,
            pool_can_draw=False,  # pool empty: a rekey to hybrid would be impossible
            active_key_tier=Action.SERVE_HYBRID,
        )
        assert mask[Action.REUSE]


def test_serve_hybrid_masked_when_pool_cannot_draw():
    request = make_request()

    cannot_draw = compute_mask(request, floor=Action.SERVE_CLASSICAL, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=False)
    assert cannot_draw[Action.SERVE_HYBRID] == False  # noqa: E712

    can_draw = compute_mask(request, floor=Action.SERVE_CLASSICAL, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=True)
    assert can_draw[Action.SERVE_HYBRID] == True  # noqa: E712


def test_nothing_masked_at_lowest_floor_pool_ok_key_fresh():
    request = make_request()
    mask = compute_mask(
        request, floor=Action.SERVE_CLASSICAL, key_age=0.0, max_key_age=MAX_KEY_AGE,
        pool_can_draw=True, active_key_tier=Action.SERVE_CLASSICAL,
    )

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
