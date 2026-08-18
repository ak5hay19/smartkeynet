"""Behavioral tests for `env.masking` (PLAN.md §4 "ACTION MASKING
(structural, inviolable)"; Hard Rule 2 -- floors are enforced by
masking, never by reward penalties, and threat signals may only raise
floors, never lower them).
"""

from __future__ import annotations

import itertools

import pytest

from env.contracts import Action, KeyType, N_ACTIONS, Request, SensitivityClass, ThreatPosture
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


# ---------------------------------------------------------------------------
# Rules 4 and 5: existing key material below the floor (2026-08-19)
#
# These close a real Hard Rule 2 hole. Rules 1-3 gate REUSE on key *age*
# only, never on the tier the existing key actually delivers, and
# REKEY_NOW refreshes the existing tier in place -- so a session could go
# on serving below-floor key material indefinitely. Measured before the
# fix on S2 (2,000 steps, seed 0, always-PQC): 275 of 1,788 REUSE
# decisions -- 15.4% -- delivered key material below the request's
# current floor, while experiments/harness.py reported floor_violations
# = 0 because it only inspected the three tier actions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key_type,floor,expected_legal",
    [
        (KeyType.PQC, Action.SERVE_HYBRID, False),      # ratcheted floor left the key behind
        (KeyType.CLASSICAL, Action.SERVE_PQC, False),
        (KeyType.CLASSICAL, Action.SERVE_HYBRID, False),
        (KeyType.HYBRID, Action.SERVE_HYBRID, True),    # exactly at the floor is fine
        (KeyType.PQC, Action.SERVE_PQC, True),
        (KeyType.HYBRID, Action.SERVE_PQC, True),       # above the floor is fine
    ],
)
def test_reuse_is_illegal_when_the_existing_key_is_below_the_floor(key_type, floor, expected_legal):
    mask = compute_mask(
        request=make_request(sensitivity_class=SensitivityClass.S3),
        floor=floor,
        key_age=0.0,           # fresh key: rule 3 cannot be what decides this
        max_key_age=500.0,
        pool_can_draw=True,
        current_key_type=key_type,
    )
    assert bool(mask[int(Action.REUSE)]) is expected_legal


@pytest.mark.parametrize(
    "key_type,floor,expected_legal",
    [
        (KeyType.PQC, Action.SERVE_HYBRID, False),
        (KeyType.CLASSICAL, Action.SERVE_PQC, False),
        (KeyType.HYBRID, Action.SERVE_HYBRID, True),
        (KeyType.PQC, Action.SERVE_CLASSICAL, True),
    ],
)
def test_rekey_now_is_illegal_when_it_would_refresh_below_the_floor(key_type, floor, expected_legal):
    """REKEY_NOW refreshes the session's *existing* tier in place
    (env/environment.py design decision 4), so it is a second below-floor
    delivery path and needs the same gate. Reachable even at flat CALM
    posture: sessions are keyed on (tenant, service) while the floor is a
    function of the *request's* sensitivity class, so two requests on one
    session can carry different floors."""
    mask = compute_mask(
        request=make_request(sensitivity_class=SensitivityClass.S2),
        floor=floor,
        key_age=0.0,
        max_key_age=500.0,
        pool_can_draw=True,
        current_key_type=key_type,
    )
    assert bool(mask[int(Action.REKEY_NOW)]) is expected_legal


def test_no_current_key_leaves_rules_4_and_5_inert():
    """`current_key_type=None` means no key established yet -- there is
    nothing to deliver below the floor, and the default preserves every
    pre-existing call shape exactly."""
    with_key_type = compute_mask(
        request=make_request(), floor=Action.SERVE_PQC, key_age=10.0,
        max_key_age=500.0, pool_can_draw=True, current_key_type=None,
    )
    without_argument = compute_mask(
        request=make_request(), floor=Action.SERVE_PQC, key_age=10.0,
        max_key_age=500.0, pool_can_draw=True,
    )
    assert list(with_key_type) == list(without_argument)


def test_a_below_floor_session_always_retains_a_legal_upgrade_path():
    """Rules 4 and 5 must never strand a request with an empty mask when
    the pool can cover the floor -- the point is to force an upgrade, not
    to deadlock. (When the pool *cannot* cover a hybrid floor, an empty
    mask is correct and env/environment.py defers -- Hard Rule 9.)"""
    for key_type in KeyType:
        for floor in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID):
            mask = compute_mask(
                request=make_request(), floor=floor, key_age=0.0, max_key_age=500.0,
                pool_can_draw=True, current_key_type=key_type,
            )
            assert mask.any(), f"empty mask for key_type={key_type}, floor={floor}"
            for action in Action:
                if mask[int(action)] and action in (
                    Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID
                ):
                    assert int(action) >= int(floor)
