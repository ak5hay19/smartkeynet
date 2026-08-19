"""Golden regression fixtures (SMARTKEYNET_BUILD_SPEC.md §S2 exit criterion:
"Golden fixture committed under `tests/golden/`").

A golden test does not assert that a number is *right* -- it asserts that a
number has not changed without someone deciding it should. That is a different
and complementary guarantee to the behavioural tests: those pin properties
("the queue drains", "no floor violations"), this pins the actual trajectory,
so a refactor that preserves every property but silently perturbs the
dynamics is still caught.

It earned its place immediately. Generating the first fixture surfaced that
the environment's *second* deferral site -- masking gap #2 in
`_prepare_decision` -- was appending to the regret log without emitting the
matching `defer_onset` event, so the §4.4 event log under-reported regret by
41% (34 regret-log entries against 20 events). Every log-based consumer,
including the dashboard, would have quietly understated the project's headline
metric. No property test would have noticed, because both numbers were
internally consistent with themselves.

If one of these assertions fails, do not edit the fixture to match. Work out
which change moved it and whether that change was intended; only then
regenerate.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from env.contracts import Action
from env.environment import SmartKeyNetEnv

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tests" / "golden"


def _load_golden(name: str) -> dict:
    with open(GOLDEN_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def _replay(config_overrides: dict) -> tuple[SmartKeyNetEnv, list[int]]:
    """Re-run the exact rollout the fixture was generated from."""
    with open(REPO / "configs" / "default.yaml", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    seed = config_overrides["seed"]
    env = SmartKeyNetEnv(
        {
            **base,
            "scenario": config_overrides["scenario"],
            "max_steps": config_overrides["max_steps"],
            "scenario_steps": config_overrides["max_steps"] + 200,
            "seed": seed,
        }
    )
    state, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    actions: list[int] = []
    for _ in range(config_overrides["max_steps"]):
        action = Action(int(rng.choice(np.flatnonzero(info["action_mask"]))))
        actions.append(int(action))
        state, _reward, _terminated, truncated, info = env.step(action)
        if truncated:
            break
    return env, actions


def test_s3_random_rollout_matches_the_golden_fixture():
    golden = _load_golden("s3_random_seed4242.json")
    env, actions = _replay(golden["config"])

    action_digest = hashlib.sha256(json.dumps(actions).encode()).hexdigest()[:16]
    assert action_digest == golden["actions_sha_prefix"], (
        "the action sequence changed -- either the mask or the RNG stream moved"
    )

    assert env._decision_count == golden["n_decisions"]
    assert len(env._regret_log) == golden["regret_events"]
    assert len(env._forced_rekey_log) == golden["forced_rekeys"]
    assert env.pool_overflow_keys == golden["pool_overflow_keys"]
    assert env._pool_sim.level == golden["final_pool_keys"]
    assert len(env._deferral_queue) == golden["final_queue_depth"]


def test_golden_event_log_counts_match():
    golden = _load_golden("s3_random_seed4242.json")
    env, _actions = _replay(golden["config"])
    counts = dict(collections.Counter(event["type"] for event in env.event_log.events))
    assert counts == golden["event_type_counts"]


def test_golden_reward_terms_match():
    """Per-term, not just the total: two terms can move in opposite directions
    and leave the total untouched, which is exactly the kind of change that
    should not pass silently."""
    golden = _load_golden("s3_random_seed4242.json")
    env, _actions = _replay(golden["config"])
    for term, expected in golden["reward_terms"].items():
        assert env.reward_terms_total[term] == pytest.approx(expected, abs=1e-4), (
            f"reward term `{term}` moved: {env.reward_terms_total[term]} vs {expected}"
        )


def test_every_regret_event_emits_exactly_one_defer_onset():
    """The invariant the fixture caught being violated.

    `defer_onset` **is** the regret event (§4.4), so the two counters must
    agree exactly -- in both directions. Too few means the log understates the
    headline metric; too many means §S2 test 2's "once per request, not once
    per waiting step" miscount has crept back in.
    """
    golden = _load_golden("s3_random_seed4242.json")
    env, _actions = _replay(golden["config"])
    onsets = sum(1 for event in env.event_log.events if event["type"] == "defer_onset")
    assert onsets == len(env._regret_log)
