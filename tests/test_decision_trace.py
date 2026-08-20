"""Behavioral tests for `env.decision_trace` -- the Explain Decision
panel's six-step trace (PLAN2 §7.3, Hard Rule 10).

Hard Rule 10 is the whole point of this module, so most of these tests
are about what the trace may and may not contain: only values the
pipeline actually computed, templated deterministically, never
narrated.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from agents.baselines import AlwaysPQCPolicy
from env.contracts import Action, KeyType, N_ACTIONS, Request, SensitivityClass
from env.decision_trace import build_decision_trace, illegal_reason
from env.environment import SmartKeyNetEnv
from env.masking import compute_mask


def load_test_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def _request() -> Request:
    return Request(
        request_id="r0", step=0, tenant="t", service="svc",
        sensitivity_class=2, pqc_capable=True, hybrid_mandatory=False,
    )


# ---------------------------------------------------------------------------
# The reasons must never disagree with the mask that was applied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "floor,key_age,pool_can_draw,current_key_type",
    list(
        itertools.product(
            [Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID],
            [0.0, 499.0, 500.0],
            [True, False],
            [None, KeyType.CLASSICAL, KeyType.PQC, KeyType.HYBRID],
        )
    ),
)
def test_illegal_reason_agrees_with_compute_mask_on_every_combination(
    floor, key_age, pool_can_draw, current_key_type
):
    """A displayed reason that disagreed with the mask actually applied
    would be exactly the "trust me" storytelling Hard Rule 10 forbids."""
    mask = compute_mask(
        request=_request(), floor=floor, key_age=key_age, max_key_age=500.0,
        pool_can_draw=pool_can_draw, current_key_type=current_key_type,
    )
    for action in Action:
        reason = illegal_reason(
            action, floor, key_age, 500.0, pool_can_draw, current_key_type
        )
        assert bool(mask[int(action)]) == (reason is None), (
            f"{action.name}: mask says legal={bool(mask[int(action)])} but reason={reason!r}"
        )


# ---------------------------------------------------------------------------
# Trace structure and content
# ---------------------------------------------------------------------------


def _first_trace(scenario: str = "S1", seed: int = 0):
    config = load_test_config(overrides={"scenario": scenario, "max_steps": 50})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=seed)
    mask = info["action_mask"]
    action = AlwaysPQCPolicy().act(state, mask)
    return build_decision_trace(env, state, mask, action), env, state, mask, action


def test_trace_has_exactly_the_six_steps_plan2_specifies():
    trace, *_ = _first_trace()
    assert [step.index for step in trace.steps] == [1, 2, 3, 4, 5, 6]
    assert [step.title for step in trace.steps] == [
        "Threat signal ingested",
        "Posture classified",
        "Policy floor lookup",
        "Action mask computed",
        "Cost comparison among legal actions",
        "Final decision",
    ]


def test_step_three_floor_always_matches_the_floor_the_env_actually_used():
    """Addition D's stated unit test: "the decision-trace assembler's
    step-3 floor lookup always matches `PolicyTable.floor()`'s actual
    return value for the same inputs (no drift between what's displayed
    and what's real)"."""
    config = load_test_config(overrides={"scenario": "S2", "max_steps": 400})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)
    policy = AlwaysPQCPolicy()

    truncated = False
    while not truncated:
        mask = info["action_mask"]
        action = policy.act(state, mask)
        trace = build_decision_trace(env, state, mask, action)
        assert trace.policy_floor == env._current_floor.name
        assert trace.policy_floor == Action(state["policy_floor"]).name
        state, _r, _t, truncated, info = env.step(action)


def test_step_four_mask_is_the_mask_the_agent_was_handed():
    trace, _env, _state, mask, _action = _first_trace()
    assert trace.mask == [bool(mask[i]) for i in range(N_ACTIONS)]


def test_trace_never_reports_a_cost_for_an_illegal_action():
    """Step 5 compares "whatever remains legal after step 4" -- listing a
    cost for a masked-out action would imply a comparison that never
    happened."""
    trace, _env, _state, mask, _action = _first_trace()
    for name in trace.action_costs:
        assert mask[int(Action[name])]


def test_step_five_degrades_honestly_when_only_one_action_was_legal():
    """PLAN2 §7.3's closing requirement, verbatim: "if a decision was
    purely floor-driven (only one legal tier existed), step 5 should say
    so plainly rather than presenting a comparison that didn't actually
    happen"."""
    config = load_test_config(overrides={"scenario": "S1", "max_steps": 2000})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)
    policy = AlwaysPQCPolicy()

    saw_single_option = False
    truncated = False
    while not truncated:
        mask = info["action_mask"]
        action = policy.act(state, mask)
        if int(np.asarray(mask, dtype=bool).sum()) == 1:
            trace = build_decision_trace(env, state, mask, action)
            saw_single_option = True
            assert trace.cost_tradeoff_existed is False
            assert "no cost tradeoff existed" in trace.steps[4].summary
            assert trace.decided_by == "floor"
        state, _r, _t, truncated, info = env.step(action)

    assert saw_single_option, "no single-legal-action decision occurred to exercise this path"


def test_step_six_says_plainly_when_a_learned_preference_decided():
    """The other honest-degradation case: several options were legal and
    the policy picked a non-cheapest one. The trace must say that rather
    than claim a rule decided it."""
    trace, env, state, mask, _action = _first_trace()
    legal = [a for a in Action if mask[int(a)]]
    assert len(legal) > 1

    costs = trace.action_costs
    cheapest = min(legal, key=lambda a: costs[a.name]["latency"] + costs[a.name]["energy"])
    non_cheapest = next((a for a in legal if a is not cheapest), None)
    assert non_cheapest is not None

    other = build_decision_trace(env, state, mask, non_cheapest)
    assert other.decided_by == "policy_preference"
    assert "learned preference" in other.steps[5].summary


def test_every_number_in_a_summary_comes_from_the_traces_own_values():
    """Hard Rule 10's core constraint, checked structurally: each step's
    summary may only contain numbers present in that step's `values`
    (or derived from them by formatting)."""
    trace, *_ = _first_trace()
    import re

    for step in trace.steps:
        flat = json_numbers(step.values)
        for token in re.findall(r"\d+\.\d+", step.summary):
            value = float(token)
            assert any(abs(value - candidate) < 5e-3 for candidate in flat), (
                f"step {step.index} summary contains {value} which is not in its values"
            )


def json_numbers(obj: Any) -> list[float]:
    """Every float reachable inside a step's `values`."""
    out: list[float] = []
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(json_numbers(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            out.extend(json_numbers(value))
    return out


def test_trace_serializes_to_plain_json_types():
    trace, *_ = _first_trace()
    import json

    payload = json.loads(json.dumps(trace.to_dict()))
    assert payload["policy_floor"] in {a.name for a in Action}
    assert len(payload["steps"]) == 6


def test_q_values_are_included_only_when_supplied():
    trace, env, state, mask, action = _first_trace()
    assert "q_values" not in trace.steps[5].values

    with_q = build_decision_trace(env, state, mask, action, q_values=[0.1] * N_ACTIONS)
    assert "q_values" in with_q.steps[5].values
    assert set(with_q.steps[5].values["q_values"]) == {a.name for a in Action}
