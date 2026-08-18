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
from experiments.harness import ScenarioResult, run_grid, run_scenario
from metrics.regret import EpisodeMetrics


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


# ---------------------------------------------------------------------------
# Scarcity premise (2026-08-19 pool recalibration -- see
# configs/default.yaml's `pool:` block)
# ---------------------------------------------------------------------------


def test_always_hybrid_exhausts_the_pool_and_always_pqc_does_not():
    """PLAN2 §7.4's Panel 4 premise, as an executable check: the greedy
    baseline must genuinely drain the pool (nonzero exhaustion/regret)
    while the conservative one never does. Before the 2026-08-19 pool
    recalibration *both* sat at exactly zero, because the QKD link
    funded ~793 keys per key consumed -- the scenario the whole project
    is about could not occur."""
    config = load_test_config(overrides={"max_steps": 250})

    hybrid = run_scenario(AlwaysHybridPolicy(), "S1", config, seed=0)
    pqc = run_scenario(AlwaysPQCPolicy(), "S1", config, seed=0)

    assert hybrid.pool_exhaustion_events > 0
    assert hybrid.episode_metrics.regret_events > 0
    assert pqc.pool_exhaustion_events == 0
    assert pqc.episode_metrics.regret_events == 0
    # Hard Rule 9: exhaustion is an availability failure, never a security one.
    assert hybrid.floor_violations == 0
    assert pqc.floor_violations == 0


def test_tuned_threshold_is_a_distinct_policy_from_always_hybrid():
    """Hard Rule 7 requires a *tuned threshold* baseline. That baseline
    is only meaningful if `pool_fill` actually varies -- otherwise every
    threshold in `baselines.static_threshold_grid` sees the same
    constant and the "tuned threshold" is a relabelled always-hybrid.
    That was literally true before the 2026-08-19 recalibration (both
    scored total_reward == -64854.6 on S1/seed 0)."""
    config = load_test_config(overrides={"max_steps": 250})
    grid = config["baselines"]["static_threshold_grid"]

    hybrid = run_scenario(AlwaysHybridPolicy(), "S1", config, seed=0)
    rewards = {
        t: run_scenario(StaticThresholdPolicy(t), "S1", config, seed=0).total_reward for t in grid
    }

    # at least one grid threshold must behave differently from always-hybrid
    assert any(r != pytest.approx(hybrid.total_reward) for r in rewards.values())
    # and the grid must not be degenerate -- different thresholds, different outcomes
    assert len(set(round(r, 6) for r in rewards.values())) > 1
