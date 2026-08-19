"""Behavioral tests for `env.masking` (PLAN.md §4 "ACTION MASKING
(structural, inviolable)"; Hard Rule 2 -- floors are enforced by
masking, never by reward penalties, and threat signals may only raise
floors, never lower them).
"""

from __future__ import annotations

import itertools
import pathlib

import pytest

from env.contracts import N_ACTIONS, Action, Request, SensitivityClass, ThreatPosture
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
        request,
        floor=Action.SERVE_CLASSICAL,
        key_age=0.0,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=True,
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
        request,
        floor=Action.SERVE_CLASSICAL,
        key_age=0.0,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=True,
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
    mask = compute_mask(
        request, floor=Action.SERVE_HYBRID, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=True
    )

    assert mask[Action.SERVE_CLASSICAL] == False  # noqa: E712
    assert mask[Action.SERVE_PQC] == False  # noqa: E712
    assert mask[Action.SERVE_HYBRID] == True  # noqa: E712


def test_reuse_masked_when_key_age_at_or_over_cap():
    request = make_request()

    over_cap = compute_mask(
        request,
        floor=Action.SERVE_CLASSICAL,
        key_age=MAX_KEY_AGE,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=True,
        active_key_tier=Action.SERVE_CLASSICAL,
    )
    assert over_cap[Action.REUSE] == False  # noqa: E712

    under_cap = compute_mask(
        request,
        floor=Action.SERVE_CLASSICAL,
        key_age=MAX_KEY_AGE - 1,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=True,
        active_key_tier=Action.SERVE_CLASSICAL,
    )
    assert under_cap[Action.REUSE] == True  # noqa: E712


# ---------------------------------------------------------------------------
# REUSE vs the floor (spec §S4 masking rule 4, implemented 2026-08-15)
# ---------------------------------------------------------------------------


def test_reuse_masked_when_there_is_no_active_key():
    request = make_request()
    mask = compute_mask(
        request,
        floor=Action.SERVE_CLASSICAL,
        key_age=0.0,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=True,
        active_key_tier=None,
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
            request,
            floor=floor,
            key_age=0.0,
            max_key_age=MAX_KEY_AGE,
            pool_can_draw=True,
            active_key_tier=key_tier,
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
            request,
            floor=floor,
            key_age=0.0,
            max_key_age=MAX_KEY_AGE,
            pool_can_draw=False,  # pool empty: a rekey to hybrid would be impossible
            active_key_tier=Action.SERVE_HYBRID,
        )
        assert mask[Action.REUSE]


def test_serve_hybrid_masked_when_pool_cannot_draw():
    request = make_request()

    cannot_draw = compute_mask(
        request,
        floor=Action.SERVE_CLASSICAL,
        key_age=0.0,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=False,
    )
    assert cannot_draw[Action.SERVE_HYBRID] == False  # noqa: E712

    can_draw = compute_mask(
        request,
        floor=Action.SERVE_CLASSICAL,
        key_age=0.0,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=True,
    )
    assert can_draw[Action.SERVE_HYBRID] == True  # noqa: E712


def test_nothing_masked_at_lowest_floor_pool_ok_key_fresh():
    request = make_request()
    mask = compute_mask(
        request,
        floor=Action.SERVE_CLASSICAL,
        key_age=0.0,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=True,
        active_key_tier=Action.SERVE_CLASSICAL,
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
        mask = compute_mask(
            request,
            floor=floor,
            key_age=key_age,
            max_key_age=MAX_KEY_AGE,
            pool_can_draw=pool_can_draw,
        )
        assert mask.any(), (
            f"deadlock at floor={floor}, key_age={key_age}, pool_can_draw={pool_can_draw}"
        )
        assert mask.shape == (N_ACTIONS,)


def test_serve_hybrid_masked_by_pool_even_when_it_is_the_floor():
    """Hard Rule 9: pool exhaustion never causes a silent downgrade --
    if floor is SERVE_HYBRID and the pool can't cover it, SERVE_HYBRID
    is masked (routing to the deferral queue), not swapped for a lower
    tier."""
    request = make_request(sensitivity_class=3, hybrid_mandatory=True)
    mask = compute_mask(
        request,
        floor=Action.SERVE_HYBRID,
        key_age=0.0,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=False,
    )

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
# §S4 test 2 -- every cell carries a citation
# ---------------------------------------------------------------------------


def test_every_cell_has_a_citation():
    """SMARTKEYNET_BUILD_SPEC.md §S4: the policy table is "loaded from YAML,
    every cell carrying a citation comment", and Hard Rule 4 forbids invented
    security constants.

    These twelve cells *are* the security policy -- masking removes every
    action below whatever floor they return -- so an uncited cell is an
    invented security decision at the most load-bearing point in the system.
    Checked mechanically because spot-checking three cells says nothing about
    the fourth.
    """
    from env.masking import load_policy_table_config

    config = load_policy_table_config()
    uncited: list[str] = []

    for sensitivity_class in SensitivityClass:
        class_block = config["table"][sensitivity_class.name]
        for posture in ThreatPosture:
            cell = class_block[posture.name]
            source = cell.get("source")
            if not isinstance(source, str) or not source.strip():
                uncited.append(f"({sensitivity_class.name}, {posture.name})")

    assert not uncited, f"policy-table cells with no citation: {uncited}"

    legacy = config["legacy_endpoint_floor"]
    assert isinstance(legacy.get("source"), str) and legacy["source"].strip()


def test_policy_table_yaml_is_the_source_of_the_loaded_table():
    """The YAML must be what the code actually uses, not a parallel copy.

    This is the same defect class as `configs/constants.yaml` being linted
    while nothing loaded it: a cited table that the masking layer ignores is
    decoration.
    """
    from env.masking import _PLACEHOLDER_FLOOR_TABLE, load_policy_table_config

    config = load_policy_table_config()
    tier_names = config["tiers"]
    for sensitivity_class in SensitivityClass:
        for posture in ThreatPosture:
            expected = Action[
                tier_names[config["table"][sensitivity_class.name][posture.name]["floor"]]
            ]
            assert _PLACEHOLDER_FLOOR_TABLE[(sensitivity_class, posture)] is expected


def test_missing_policy_table_cell_raises(tmp_path):
    """A silently absent cell would fall back to some default floor, and a
    *lower* floor reached by accident is exactly the Hard Rule 2 failure this
    layer exists to prevent."""
    import yaml as _yaml

    from env.masking import _build_floor_table, load_policy_table_config

    config = load_policy_table_config()
    del config["table"]["S3"]["HIGH"]
    with pytest.raises(ValueError, match="S3"):
        _build_floor_table(config)


# ---------------------------------------------------------------------------
# §S4 rule 4 last line -- min_rekey_interval; §S2 -- head-of-line reservation
# ---------------------------------------------------------------------------


def _plain_request(hybrid_mandatory: bool = False) -> dict:
    return {
        "request_id": "r0",
        "tenant": "hospital",
        "service": "svc",
        "sensitivity_class": 3,
        "pqc_capable": True,
        "hybrid_mandatory": hybrid_mandatory,
    }


def test_rekey_masked_inside_the_minimum_interval():
    """§S4 masking rule 4, last line. §7.3 names "rekey thrashing" as a
    degenerate policy whose causes are "`c_rekey_base` too low, or
    `min_rekey_interval` unset" -- this closes the second, so the agent cannot
    re-establish a key every step to farm the freshness bonus."""
    fresh = compute_mask(
        request=_plain_request(),
        floor=Action.SERVE_PQC,
        key_age=2.0,
        max_key_age=500.0,
        pool_can_draw=True,
        active_key_tier=Action.SERVE_PQC,
        steps_since_rekey=2.0,
        min_rekey_interval=5.0,
    )
    assert not fresh[int(Action.REKEY_NOW)]
    assert fresh.any(), "liveness: masking the rekey must not empty the mask"

    settled = compute_mask(
        request=_plain_request(),
        floor=Action.SERVE_PQC,
        key_age=20.0,
        max_key_age=500.0,
        pool_can_draw=True,
        active_key_tier=Action.SERVE_PQC,
        steps_since_rekey=20.0,
        min_rekey_interval=5.0,
    )
    assert settled[int(Action.REKEY_NOW)]


def test_min_rekey_interval_defaults_to_no_restriction():
    """Zero interval must behave exactly as before the feature existed, so
    every recorded result stays interpretable."""
    mask = compute_mask(
        request=_plain_request(),
        floor=Action.SERVE_PQC,
        key_age=1.0,
        max_key_age=500.0,
        pool_can_draw=True,
        active_key_tier=Action.SERVE_PQC,
    )
    assert mask[int(Action.REKEY_NOW)]


def test_strict_head_reservation_masks_discretionary_hybrid_while_queue_waits():
    """§S2's `queue.head_reservation: strict`.

    A discretionary hybrid serve must not consume the key a queued
    higher-class request is waiting for. The default is `none` because -- in
    the spec's words -- "the agent *must be able to make the mistake* for the
    regret metric to mean anything".
    """
    discretionary = compute_mask(
        request=_plain_request(hybrid_mandatory=False),
        floor=Action.SERVE_CLASSICAL,
        key_age=10.0,
        max_key_age=500.0,
        pool_can_draw=True,
        active_key_tier=Action.SERVE_PQC,
        queue_non_empty=True,
        head_reservation="strict",
        request_is_hybrid_mandatory=False,
    )
    assert not discretionary[int(Action.SERVE_HYBRID)]
    assert discretionary.any()


def test_strict_head_reservation_never_blocks_a_mandatory_request():
    """The reservation exists to protect mandatory requests, so it must never
    be the thing that starves one -- that would invert its purpose and could
    empty the mask for a hybrid-floored request."""
    mandatory = compute_mask(
        request=_plain_request(hybrid_mandatory=True),
        floor=Action.SERVE_HYBRID,
        key_age=10.0,
        max_key_age=500.0,
        pool_can_draw=True,
        active_key_tier=Action.SERVE_HYBRID,
        queue_non_empty=True,
        head_reservation="strict",
        request_is_hybrid_mandatory=True,
    )
    assert mandatory[int(Action.SERVE_HYBRID)]
    assert mandatory.any()


def test_default_head_reservation_allows_the_queue_jump():
    """The mistake must remain reachable under the default, or `regret_events`
    measures only environmental scarcity and says nothing about the policy."""
    mask = compute_mask(
        request=_plain_request(hybrid_mandatory=False),
        floor=Action.SERVE_CLASSICAL,
        key_age=10.0,
        max_key_age=500.0,
        pool_can_draw=True,
        active_key_tier=Action.SERVE_PQC,
        queue_non_empty=True,
        head_reservation="none",
    )
    assert mask[int(Action.SERVE_HYBRID)]


def test_env_rejects_an_unknown_head_reservation_mode():
    import yaml as _yaml

    from env.environment import SmartKeyNetEnv

    config = _yaml.safe_load(
        (pathlib.Path(__file__).resolve().parent.parent / "configs" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["queue"] = {**config.get("queue", {}), "head_reservation": "sometimes"}
    with pytest.raises(ValueError, match="head_reservation"):
        SmartKeyNetEnv({**config, "scenario": "S1", "max_steps": 5})
