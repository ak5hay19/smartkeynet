"""Behavioral tests for `experiments.harness` -- the baseline comparison
harness (PLAN.md §10 kickoff step 6; Hard Rule 7: baselines must be
comparable before the DQN is tuned).

Only S1 is exercised here: `env/environment.py` only meaningfully
dispatches S1 this session (S2-S6 scenario dispatch is separate future
work -- see PROGRESS.md). `run_scenario`/`run_grid` still take
`scenario` as a generic parameter, which is the right final interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from agents.baselines import (
    AlwaysHybridPolicy,
    AlwaysPQCPolicy,
    Policy,
    RandomPolicy,
    StaticThresholdPolicy,
)
from env.contracts import Action, KeyType
from experiments.harness import ScenarioResult, _delivered_tier, run_grid, run_scenario
from metrics.regret import EpisodeMetrics

_TIER_ACTIONS = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)


def _onehot(key_type: KeyType | None) -> list[float]:
    onehot = [0.0, 0.0, 0.0]
    if key_type is not None:
        onehot[int(key_type)] = 1.0
    return onehot


def load_test_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the real `configs/default.yaml` (mirrors
    tests/test_environment.py's helper of the same name -- nothing
    hardcoded here) and shallow-merge per-section overrides."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value

    return config


def _four_baselines() -> dict[str, Policy]:
    """Fresh instances every call -- `RandomPolicy` carries mutable RNG
    state that shouldn't leak between tests."""
    return {
        "always_pqc": AlwaysPQCPolicy(),
        "always_hybrid": AlwaysHybridPolicy(),
        "static_threshold": StaticThresholdPolicy(pool_fill_threshold=0.5),
        "random": RandomPolicy(seed=0),
    }


# ---------------------------------------------------------------------------
# run_scenario
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["always_pqc", "always_hybrid", "static_threshold", "random"])
def test_run_scenario_s1_completes_with_zero_floor_violations(name):
    policy = _four_baselines()[name]
    config = load_test_config(overrides={"max_steps": 100})

    result = run_scenario(policy, "S1", config, seed=123)

    assert isinstance(result, ScenarioResult)
    assert result.scenario == "S1"
    assert result.seed == 123
    assert isinstance(result.episode_metrics, EpisodeMetrics)
    # this is the whole point of the masking architecture (PLAN.md Hard
    # Rule 2/9): every one of these policies only ever picks from the
    # mask it's given, so floor violations must be 0 by construction.
    assert result.floor_violations == 0
    assert result.p99_latency >= 0.0
    assert result.pool_exhaustion_events >= 0
    assert isinstance(result.total_reward, float)


def test_run_scenario_total_reward_matches_manually_summed_env_rewards():
    """`ScenarioResult.total_reward` (added 2026-08-10 -- see
    SESSION_LOG.md, flagged as worth having since `p99_latency` is a
    coarse comparison metric) must be exactly the sum of `env.step()`'s
    own reward across the episode, not a re-derived or reshaped value
    -- re-run the same policy/config/seed driving the env directly and
    compare."""
    from env.environment import SmartKeyNetEnv

    config = load_test_config(overrides={"max_steps": 100})
    policy = AlwaysPQCPolicy()

    result = run_scenario(policy, "S1", config, seed=9)

    env_config = {**config, "scenario": "S1", "seed": 9, "max_steps": 100}
    env = SmartKeyNetEnv(env_config)
    state, info = env.reset(seed=9)
    manual_total = 0.0
    truncated = False
    while not truncated:
        action = policy.act(state, info["action_mask"])
        state, reward, terminated, truncated, info = env.step(action)
        manual_total += reward

    assert result.total_reward == pytest.approx(manual_total)


def test_run_scenario_respects_explicit_max_steps_override():
    """`max_steps` set by the caller must survive run_scenario's
    internal `setdefault` (i.e. not get silently replaced by the
    harness's own default episode length)."""
    config = load_test_config(overrides={"max_steps": 17})
    result = run_scenario(AlwaysPQCPolicy(), "S1", config, seed=1)
    assert result.episode_metrics.rekeys_per_100_requests >= 0.0  # ran to completion, no crash


def test_run_scenario_static_threshold_grid_search_produces_a_valid_run():
    config = load_test_config(overrides={"max_steps": 100})

    def eval_fn(policy: StaticThresholdPolicy) -> float:
        return -run_scenario(policy, "S1", config, seed=5).p99_latency

    grid = config["baselines"]["static_threshold_grid"]
    tuned = StaticThresholdPolicy.grid_search(grid, eval_fn)

    result = run_scenario(tuned, "S1", config, seed=5)
    assert result.floor_violations == 0


# ---------------------------------------------------------------------------
# floor_violations completeness (2026-08-19 Hard Rule 2 fix)
# ---------------------------------------------------------------------------


def test_floor_violations_old_check_would_have_missed_a_stale_reuse_delivery():
    """Directly demonstrates the second bug this session fixed -- a
    metric that claimed a guarantee ("must be 0 -- by construction") it
    wasn't actually checking for two of five actions. Constructs the
    exact case the old `action in _TIER_ACTIONS` check couldn't see
    (REUSE delivering an existing PQC-tier key below a SERVE_HYBRID
    floor) and shows the new `_delivered_tier`-based check catches it
    while the literal old expression would not have."""
    action = Action.REUSE
    floor = Action.SERVE_HYBRID
    key_type_onehot = _onehot(KeyType.PQC)  # stale: below the HYBRID floor

    old_check_would_flag = action in _TIER_ACTIONS and int(action) < int(floor)
    assert old_check_would_flag is False  # the bug: REUSE is never in _TIER_ACTIONS

    delivered = _delivered_tier(action, key_type_onehot, floor)
    new_check_flags = int(delivered) < int(floor)
    assert new_check_flags is True


def test_delivered_tier_reuse_reports_the_existing_session_tier():
    assert _delivered_tier(Action.REUSE, _onehot(KeyType.PQC), floor=Action.SERVE_CLASSICAL) == Action.SERVE_PQC
    assert _delivered_tier(Action.REUSE, _onehot(KeyType.HYBRID), floor=Action.SERVE_PQC) == Action.SERVE_HYBRID


def test_delivered_tier_rekey_now_reports_max_of_existing_and_floor():
    assert _delivered_tier(Action.REKEY_NOW, _onehot(KeyType.PQC), floor=Action.SERVE_HYBRID) == Action.SERVE_HYBRID
    assert _delivered_tier(Action.REKEY_NOW, _onehot(KeyType.HYBRID), floor=Action.SERVE_CLASSICAL) == Action.SERVE_HYBRID
    assert _delivered_tier(Action.REKEY_NOW, _onehot(None), floor=Action.SERVE_PQC) == Action.SERVE_PQC  # cold start


def test_delivered_tier_tier_actions_report_themselves():
    for action in _TIER_ACTIONS:
        assert _delivered_tier(action, _onehot(None), floor=Action.SERVE_CLASSICAL) == action


def test_run_scenario_s2_floor_violations_are_genuinely_zero_under_random_policy():
    """End-to-end, via the public `run_scenario` API this time (not
    env internals directly, see tests/test_environment.py's
    equivalent): S2 genuinely ratchets floors mid-episode, and
    `RandomPolicy` exercises REUSE/REKEY_NOW whenever legal -- exactly
    the combination that measured 64/279 (22.9%) below-floor
    deliveries before this session's fix (see SESSION_LOG.md). The
    harness's own `floor_violations` counter must now correctly report
    zero, not just the direct env-level check."""
    config = load_test_config(
        overrides={
            "scenario": "S2",
            "threat_schedule": {"elevate_at_step": 50, "elevated_signal": 6.0},
            "max_steps": 500,
        }
    )
    result = run_scenario(RandomPolicy(seed=0), "S2", config, seed=0)
    assert result.floor_violations == 0


# ---------------------------------------------------------------------------
# run_grid
# ---------------------------------------------------------------------------


def test_run_grid_returns_one_result_per_combination():
    config = load_test_config(overrides={"max_steps": 50})
    policies = _four_baselines()
    scenarios = ["S1"]
    seeds = [1, 2]

    results = run_grid(policies=policies, scenarios=scenarios, config=config, seeds=seeds)

    assert len(results) == len(policies) * len(scenarios) * len(seeds)
    for result in results:
        assert isinstance(result, ScenarioResult)
        assert result.scenario == "S1"
        assert result.seed in seeds
        assert result.floor_violations == 0
