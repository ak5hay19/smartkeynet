"""Behavioral tests for `dashboard.explain` (PLAN2.md §7.3, Addition D;
Hard Rule 10 -- the Explain Decision panel may only display values the
pipeline actually computed, never a synthesized explanation).

These tests check the module's central claim directly: that step 3's
floor and step 4's per-action legal/reason fields cannot drift from
`env/masking.py`'s real `PolicyTable`/`compute_mask()` behavior, that
step 5's costs come from `env/environment.py`'s real cost tables (not
re-typed numbers), and that step 6's final sentence is deterministic
and only ever quotes values that appear elsewhere in the same trace.
"""

from __future__ import annotations

import itertools

import pytest

from dashboard.explain import DecisionTrace, explain_decision, explain_decision_from_env
from env.contracts import Action, KeyType, Request, SensitivityClass, ThreatPosture
from env.environment import _ENERGY_UNITS, _LATENCY_UNITS, SmartKeyNetEnv
from env.masking import PolicyTable, compute_mask, load_key_lifetime_config

MAX_KEY_AGE = load_key_lifetime_config()["max_key_age_steps"]

_COLD_START_ONEHOT = [0.0, 0.0, 0.0]


def _onehot(key_type: KeyType | None) -> list[float]:
    onehot = [0.0, 0.0, 0.0]
    if key_type is not None:
        onehot[int(key_type)] = 1.0
    return onehot


def make_request(sensitivity_class: int = 0, hybrid_mandatory: bool = False) -> Request:
    return Request(
        request_id="r0",
        step=0,
        tenant="hospital",
        service="export",
        sensitivity_class=sensitivity_class,
        pqc_capable=True,
        hybrid_mandatory=hybrid_mandatory,
    )


def _default_kwargs(**overrides):
    kwargs = dict(
        request=make_request(sensitivity_class=0),
        threat_score=0.1,
        threat_source="test",
        posture_probs=None,
        floor=Action.SERVE_CLASSICAL,
        key_age=0.0,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=True,
        key_type_onehot=_COLD_START_ONEHOT,
        chosen_action=Action.SERVE_CLASSICAL,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Step 3: floor lookup must match env/masking.py's real floor table
# ---------------------------------------------------------------------------


def test_floor_table_matches_masking_py_across_every_cell():
    table = PolicyTable()
    trace = explain_decision(**_default_kwargs())

    for sensitivity_class, posture in itertools.product(SensitivityClass, ThreatPosture):
        assert trace.floor_table[(sensitivity_class, posture)] == table.floor(sensitivity_class, posture)


def test_trace_floor_echoes_the_real_policy_table_lookup_for_a_spread_of_cells():
    for sensitivity_class, posture in itertools.product(SensitivityClass, ThreatPosture):
        table = PolicyTable()  # fresh table per case -- no ratchet carryover
        real_floor = table.floor(sensitivity_class, posture)
        # build posture_probs whose argmax is exactly `posture`, so step 2's
        # resolved posture (and hence floor_cell) matches this cell
        probs = [0.01, 0.01, 0.01]
        probs[int(posture)] = 1.0

        trace = explain_decision(
            **_default_kwargs(
                request=make_request(sensitivity_class=int(sensitivity_class)),
                posture_probs=probs,
                floor=real_floor,
                chosen_action=real_floor,  # the floor tier is always legal at its own cell
            )
        )
        assert trace.floor == real_floor
        assert trace.floor_cell == (sensitivity_class, posture)


# ---------------------------------------------------------------------------
# Step 4: mask legal/reason must match compute_mask()'s real boolean output
# ---------------------------------------------------------------------------


def _real_mask(request, floor, key_age, max_key_age, pool_can_draw):
    return compute_mask(request=request, floor=floor, key_age=key_age, max_key_age=max_key_age, pool_can_draw=pool_can_draw)


@pytest.mark.parametrize(
    "floor,key_age,pool_can_draw,key_type_onehot",
    [
        # pool empty
        (Action.SERVE_CLASSICAL, 0.0, False, _COLD_START_ONEHOT),
        # key age at cap exactly
        (Action.SERVE_CLASSICAL, MAX_KEY_AGE, True, _onehot(KeyType.CLASSICAL)),
        # key age over cap
        (Action.SERVE_CLASSICAL, MAX_KEY_AGE + 50, True, _onehot(KeyType.PQC)),
        # cold start (no existing key) -- key_age pinned to max_key_age, per environment.py design decision 2
        (Action.SERVE_PQC, MAX_KEY_AGE, True, _COLD_START_ONEHOT),
        # all-legal case: lowest floor, pool ok, key fresh
        (Action.SERVE_CLASSICAL, 0.0, True, _onehot(KeyType.CLASSICAL)),
        # floor at HYBRID, pool cannot cover it (Hard Rule 9 case)
        (Action.SERVE_HYBRID, 0.0, False, _COLD_START_ONEHOT),
    ],
)
def test_mask_legal_flags_match_compute_mask_exactly(floor, key_age, pool_can_draw, key_type_onehot):
    request = make_request(sensitivity_class=int(floor))  # any class; floor is passed explicitly
    real_mask = _real_mask(request, floor, key_age, MAX_KEY_AGE, pool_can_draw)

    # pick any real-legal action as the chosen one (mask always has >=1 legal action)
    chosen = next(a for a in Action if bool(real_mask[int(a)]))

    trace = explain_decision(
        **_default_kwargs(
            request=request,
            floor=floor,
            key_age=key_age,
            pool_can_draw=pool_can_draw,
            key_type_onehot=key_type_onehot,
            chosen_action=chosen,
        )
    )

    entries_by_action = {e.action: e for e in trace.mask}
    for action in Action:
        assert entries_by_action[action].legal == bool(real_mask[int(action)]), (
            f"{action.name} legal flag drifted from compute_mask() at "
            f"floor={floor.name}, key_age={key_age}, pool_can_draw={pool_can_draw}"
        )


@pytest.mark.parametrize(
    "floor,key_age,pool_can_draw,key_type_onehot",
    [
        (Action.SERVE_CLASSICAL, 0.0, False, _COLD_START_ONEHOT),
        (Action.SERVE_CLASSICAL, MAX_KEY_AGE, True, _onehot(KeyType.CLASSICAL)),
        (Action.SERVE_CLASSICAL, MAX_KEY_AGE + 50, True, _onehot(KeyType.PQC)),
        (Action.SERVE_PQC, MAX_KEY_AGE, True, _COLD_START_ONEHOT),
        (Action.SERVE_CLASSICAL, 0.0, True, _onehot(KeyType.CLASSICAL)),
        (Action.SERVE_HYBRID, 0.0, False, _COLD_START_ONEHOT),
    ],
)
def test_illegal_actions_have_nonempty_reason_and_legal_actions_have_no_spurious_reason(
    floor, key_age, pool_can_draw, key_type_onehot
):
    request = make_request(sensitivity_class=int(floor))
    real_mask = _real_mask(request, floor, key_age, MAX_KEY_AGE, pool_can_draw)
    chosen = next(a for a in Action if bool(real_mask[int(a)]))

    trace = explain_decision(
        **_default_kwargs(
            request=request,
            floor=floor,
            key_age=key_age,
            pool_can_draw=pool_can_draw,
            key_type_onehot=key_type_onehot,
            chosen_action=chosen,
        )
    )

    for entry in trace.mask:
        if entry.legal:
            assert entry.reason  # every legal action still gets a positive reason
            assert "below floor" not in entry.reason
            assert "cannot cover" not in entry.reason
            assert "exceeded" not in entry.reason
        else:
            assert entry.reason  # non-empty, per the module's contract
            assert entry.reason != "illegal"


def test_cold_start_reuse_reason_says_cold_start_not_generic_age_cap():
    request = make_request(sensitivity_class=0)
    trace = explain_decision(
        **_default_kwargs(
            request=request,
            floor=Action.SERVE_CLASSICAL,
            key_age=MAX_KEY_AGE,  # cold-start sessions are pinned here by environment.py
            pool_can_draw=True,
            key_type_onehot=_COLD_START_ONEHOT,
            chosen_action=Action.SERVE_CLASSICAL,
        )
    )
    reuse_entry = next(e for e in trace.mask if e.action is Action.REUSE)
    assert reuse_entry.legal is False
    assert "cold start" in reuse_entry.reason


def test_stale_tier_reuse_reason_matches_compute_mask_and_names_the_existing_tier():
    """2026-08-19 Hard Rule 2 fix: an established key below the current
    floor (age well within cap) must show as illegal, with a reason
    naming the actual stale tier -- cross-checked against a real
    `compute_mask()` call, not a hardcoded expectation."""
    request = make_request(sensitivity_class=3)
    floor = Action.SERVE_HYBRID
    key_type_onehot = _onehot(KeyType.PQC)  # stale: below the HYBRID floor
    key_age = 0.0  # well within cap -- isolates the new rule from the age rule

    real_mask = compute_mask(
        request=request, floor=floor, key_age=key_age, max_key_age=MAX_KEY_AGE,
        pool_can_draw=True, current_key_type=KeyType.PQC,
    )
    assert real_mask[Action.REUSE] == False  # noqa: E712 -- sanity: the real function agrees this is illegal

    chosen = next(a for a in Action if bool(real_mask[int(a)]))
    trace = explain_decision(
        **_default_kwargs(
            request=request,
            floor=floor,
            key_age=key_age,
            pool_can_draw=True,
            key_type_onehot=key_type_onehot,
            chosen_action=chosen,
        )
    )

    reuse_entry = next(e for e in trace.mask if e.action is Action.REUSE)
    assert reuse_entry.legal is False
    assert "SERVE_PQC" in reuse_entry.reason
    assert "below floor" in reuse_entry.reason


def test_stale_tier_reuse_reason_does_not_fire_when_tier_still_meets_floor():
    """Regression check: an established key at/above the current floor
    must not spuriously get the new reason."""
    request = make_request(sensitivity_class=0)
    trace = explain_decision(
        **_default_kwargs(
            request=request,
            floor=Action.SERVE_PQC,
            key_age=0.0,
            pool_can_draw=True,
            key_type_onehot=_onehot(KeyType.PQC),  # exactly at floor
            chosen_action=Action.REUSE,
        )
    )
    reuse_entry = next(e for e in trace.mask if e.action is Action.REUSE)
    assert reuse_entry.legal is True
    assert "below floor" not in reuse_entry.reason


def test_rekey_now_cost_reflects_escalated_tier_not_stale_existing_tier():
    """Step 5's cost display must match what REKEY_NOW actually
    delivers post-fix -- max(existing tier, floor), not the stale
    existing tier -- otherwise the panel would show a cost for a tier
    that was never really served (a real Hard Rule 10 drift risk)."""
    trace = explain_decision(
        **_default_kwargs(
            request=make_request(sensitivity_class=3),
            floor=Action.SERVE_HYBRID,
            key_age=0.0,
            pool_can_draw=True,
            key_type_onehot=_onehot(KeyType.PQC),  # stale existing tier
            chosen_action=Action.REKEY_NOW,
        )
    )
    rekey_entry = next(c for c in trace.costs if c.action is Action.REKEY_NOW)
    assert rekey_entry.latency == _LATENCY_UNITS[Action.SERVE_HYBRID]
    assert rekey_entry.energy == _ENERGY_UNITS[Action.SERVE_HYBRID]


def test_chosen_action_must_be_legal():
    with pytest.raises(ValueError):
        explain_decision(
            **_default_kwargs(
                floor=Action.SERVE_HYBRID,
                key_age=0.0,
                pool_can_draw=False,  # HYBRID illegal
                chosen_action=Action.SERVE_HYBRID,
            )
        )


# ---------------------------------------------------------------------------
# Step 5: costs must come from env/environment.py's real cost constants
# ---------------------------------------------------------------------------


def test_costs_match_environment_py_real_constants_exactly():
    trace = explain_decision(
        **_default_kwargs(
            request=make_request(sensitivity_class=0),
            floor=Action.SERVE_CLASSICAL,
            key_age=0.0,
            pool_can_draw=True,
            key_type_onehot=_onehot(KeyType.CLASSICAL),
            chosen_action=Action.REUSE,
        )
    )
    costs_by_action = {c.action: c for c in trace.costs}
    for action in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID, Action.REUSE):
        assert costs_by_action[action].latency == _LATENCY_UNITS[action]
        assert costs_by_action[action].energy == _ENERGY_UNITS[action]


def test_rekey_now_cost_resolves_to_existing_session_tier_not_its_own_entry():
    """REKEY_NOW has no entry in `_LATENCY_UNITS`/`_ENERGY_UNITS` -- it must
    cost against whichever tier it actually refreshes."""
    trace = explain_decision(
        **_default_kwargs(
            request=make_request(sensitivity_class=0),
            floor=Action.SERVE_CLASSICAL,
            key_age=0.0,
            pool_can_draw=True,
            key_type_onehot=_onehot(KeyType.HYBRID),
            chosen_action=Action.REKEY_NOW,
        )
    )
    rekey_entry = next(c for c in trace.costs if c.action is Action.REKEY_NOW)
    assert rekey_entry.latency == _LATENCY_UNITS[Action.SERVE_HYBRID]
    assert rekey_entry.energy == _ENERGY_UNITS[Action.SERVE_HYBRID]


def test_rekey_now_cost_resolves_to_floor_tier_on_cold_start():
    trace = explain_decision(
        **_default_kwargs(
            request=make_request(sensitivity_class=1),
            floor=Action.SERVE_PQC,
            key_age=MAX_KEY_AGE,
            pool_can_draw=True,
            key_type_onehot=_COLD_START_ONEHOT,
            chosen_action=Action.REKEY_NOW,
        )
    )
    rekey_entry = next(c for c in trace.costs if c.action is Action.REKEY_NOW)
    assert rekey_entry.latency == _LATENCY_UNITS[Action.SERVE_PQC]
    assert rekey_entry.energy == _ENERGY_UNITS[Action.SERVE_PQC]


def test_only_one_legal_action_gets_a_cost_note():
    """floor=HYBRID + pool empty makes SERVE_HYBRID illegal despite being
    the floor (Hard Rule 9); key_age at cap makes REUSE illegal too;
    REKEY_NOW is the only action compute_mask()'s 3 rules never gate
    except via the floor comparison, which it always clears -- so this
    is the one combination where exactly one action is ever legal."""
    trace = explain_decision(
        **_default_kwargs(
            request=make_request(sensitivity_class=3),
            floor=Action.SERVE_HYBRID,
            key_age=MAX_KEY_AGE,
            pool_can_draw=False,
            key_type_onehot=_COLD_START_ONEHOT,
            chosen_action=Action.REKEY_NOW,
        )
    )
    assert [e.action for e in trace.costs] == [Action.REKEY_NOW]
    assert trace.cost_note is not None


# ---------------------------------------------------------------------------
# Step 6: final text must be deterministic and only quote real trace values
# ---------------------------------------------------------------------------


def test_final_text_is_deterministic():
    kwargs = _default_kwargs(
        request=make_request(sensitivity_class=1),
        floor=Action.SERVE_PQC,
        key_age=0.0,
        pool_can_draw=True,
        key_type_onehot=_COLD_START_ONEHOT,
        chosen_action=Action.SERVE_PQC,
    )
    text_a = explain_decision(**kwargs).final_text
    text_b = explain_decision(**kwargs).final_text
    assert text_a == text_b


@pytest.mark.parametrize(
    "floor,key_age,pool_can_draw,key_type_onehot,chosen_action,sensitivity_class",
    [
        (Action.SERVE_HYBRID, MAX_KEY_AGE, False, _COLD_START_ONEHOT, Action.REKEY_NOW, 3),  # floor-only (see test_only_one_legal_action_gets_a_cost_note)
        (Action.SERVE_CLASSICAL, 0.0, True, _onehot(KeyType.CLASSICAL), Action.SERVE_CLASSICAL, 0),  # cheapest chosen
        (Action.SERVE_PQC, MAX_KEY_AGE, True, _onehot(KeyType.HYBRID), Action.REKEY_NOW, 2),  # learned preference
    ],
)
def test_final_text_only_contains_values_from_the_same_trace(
    floor, key_age, pool_can_draw, key_type_onehot, chosen_action, sensitivity_class
):
    trace = explain_decision(
        **_default_kwargs(
            request=make_request(sensitivity_class=sensitivity_class),
            floor=floor,
            key_age=key_age,
            pool_can_draw=pool_can_draw,
            key_type_onehot=key_type_onehot,
            chosen_action=chosen_action,
        )
    )

    text = trace.final_text
    assert floor.name in text
    assert trace.sensitivity_class.name in text
    assert trace.resolved_posture.name in text
    assert trace.chosen_action.name in text

    # every action name mentioned in the sentence must be a real action
    # that appears somewhere in this trace's own mask/costs -- never an
    # invented one
    known_action_names = {e.action.name for e in trace.mask}
    for action in Action:
        if action.name in text:
            assert action.name in known_action_names


# ---------------------------------------------------------------------------
# End-to-end sanity: a real env decision produces a self-consistent trace
# ---------------------------------------------------------------------------


def _base_config(**overrides):
    config = {
        "scenario": "S1",
        "seed": 0,
        "use_foresight": "off",
        "pool": {"capacity_bits": 1_000_000, "initial_fill_frac": 0.5, "bits_per_hybrid_draw": 256},
        "key_lifetime": {"max_key_age_steps": MAX_KEY_AGE},
        "reward": {"w_lat": 1.0, "w_en": 0.1, "w_fr": 0.1, "w_qkd": 1.0, "r_starve": 10.0, "c_rekey_base": 1.0, "c_rekey_load_beta": 1.0},
        "max_steps": 20,
    }
    config.update(overrides)
    return config


def test_explain_decision_from_env_produces_a_self_consistent_trace_on_real_env_steps():
    env = SmartKeyNetEnv(_base_config())
    state, info = env.reset(seed=0)

    for _ in range(10):
        mask = info["action_mask"]
        chosen = next(a for a in Action if bool(mask[int(a)]))  # any legal action -- policy-agnostic

        trace = explain_decision_from_env(env, state, chosen)

        assert isinstance(trace, DecisionTrace)
        assert trace.chosen_action is chosen
        assert any(e.action is chosen and e.legal for e in trace.mask)
        # cross-check step 4 directly against a fresh compute_mask() call
        # built from the same inputs the wrapper read off the env,
        # including current_key_type (2026-08-19 fix) -- without it this
        # cross-check compares against the wrong (pre-fix) rule set.
        real_floor = Action(state["policy_floor"])
        current_key_type = next(
            (kt for kt in KeyType if state["key_type_onehot"][int(kt)] == 1.0), None
        )
        real_mask = compute_mask(
            request=env._current_request,
            floor=real_floor,
            key_age=state["key_age"],
            max_key_age=env._max_key_age,
            pool_can_draw=env._pool_sim.can_draw(env._bits_per_hybrid_draw),
            current_key_type=current_key_type,
        )
        for entry in trace.mask:
            assert entry.legal == bool(real_mask[int(entry.action)])

        state, reward, terminated, truncated, info = env.step(chosen)
        if truncated:
            break


def test_explain_decision_from_env_with_ewma_forecaster_has_real_posture_probs():
    config = _base_config(use_foresight="ewma")
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)

    mask = info["action_mask"]
    chosen = next(a for a in Action if bool(mask[int(a)]))
    trace = explain_decision_from_env(env, state, chosen)

    assert trace.posture_probs is not None
    assert abs(sum(trace.posture_probs.values()) - 1.0) < 1e-6
    assert trace.threat_source == "MovingAverageForecaster"
