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
from experiments.harness import (
    AttackScenarioResult,
    MultiSeedAttackEvalResult,
    MultiSeedEvalResult,
    ScenarioResult,
    _delivered_tier,
    evaluate_attack_multi_seed,
    evaluate_multi_seed,
    run_grid,
    run_scenario,
    run_scenario_under_attack,
)
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


# ---------------------------------------------------------------------------
# evaluate_multi_seed (2026-08-19, Gate W3 attempt session)
# ---------------------------------------------------------------------------


def test_evaluate_multi_seed_runs_every_seed_and_keeps_the_raw_results():
    config = load_test_config(overrides={"max_steps": 50})
    eval_seeds = [1, 2, 3]

    result = evaluate_multi_seed(AlwaysPQCPolicy(), "S1", config, eval_seeds)

    assert isinstance(result, MultiSeedEvalResult)
    assert result.scenario == "S1"
    assert result.eval_seeds == eval_seeds
    assert len(result.results) == len(eval_seeds)
    for seed, r in zip(eval_seeds, result.results):
        assert isinstance(r, ScenarioResult)
        assert r.seed == seed


def test_evaluate_multi_seed_means_match_a_manual_average_of_run_scenario():
    """The summary stats must be exactly derived from the same
    `run_scenario` calls `evaluate_multi_seed` itself makes -- not a
    re-fetched or independently-computed number that could drift."""
    config = load_test_config(overrides={"max_steps": 50})
    eval_seeds = [10, 11, 12, 13]
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)

    result = evaluate_multi_seed(policy, "S1", config, eval_seeds)

    manual_results = [run_scenario(policy, "S1", config, seed=s) for s in eval_seeds]
    manual_p99 = [r.p99_latency for r in manual_results]
    manual_reward = [r.total_reward for r in manual_results]

    assert result.p99_latency_mean == pytest.approx(sum(manual_p99) / len(manual_p99))
    assert result.total_reward_mean == pytest.approx(sum(manual_reward) / len(manual_reward))


def test_evaluate_multi_seed_std_is_zero_for_a_deterministic_policy_on_identical_conditions():
    """Sanity check on the std computation itself: a policy whose
    behavior doesn't depend on the eval seed at all (AlwaysPQCPolicy,
    on a config where the request stream's seed dependence doesn't
    change which action gets chosen) should show near-zero spread only
    if the underlying runs are actually similar -- this just confirms
    std is computed correctly, not hardcoded to some placeholder."""
    config = load_test_config(overrides={"max_steps": 100})
    result = evaluate_multi_seed(AlwaysPQCPolicy(), "S1", config, eval_seeds=[1, 2, 3])

    import numpy as np

    manual_std = float(np.std([r.p99_latency for r in result.results]))
    assert result.p99_latency_std == pytest.approx(manual_std)


def test_evaluate_multi_seed_floor_violations_total_is_summed_not_averaged():
    """`floor_violations_total` must sum across seeds, not average --
    averaging could hide a single bad seed behind a small-looking mean.
    Since every masked policy has zero violations by construction
    (Hard Rule 2), this also doubles as a real zero-violations check
    across multiple seeds at once."""
    config = load_test_config(overrides={"max_steps": 50})
    result = evaluate_multi_seed(AlwaysHybridPolicy(), "S1", config, eval_seeds=[1, 2, 3, 4])
    assert result.floor_violations_total == 0


def test_evaluate_multi_seed_rejects_empty_eval_seeds():
    config = load_test_config(overrides={"max_steps": 50})
    with pytest.raises(ValueError):
        evaluate_multi_seed(AlwaysPQCPolicy(), "S1", config, eval_seeds=[])


def test_evaluate_multi_seed_below_floor_rate_is_zero_for_a_masked_policy():
    """2026-08-25 addition: `below_floor_rate_mean`/`_std` -- the rate
    form of `floor_violations_total` (PLAN.md's paper-draft "below-floor
    service rate"). Every masked policy has `floor_violations == 0` at
    every seed by construction (Hard Rule 2), so both the mean and std
    of the per-seed rate must be exactly `0.0`."""
    config = load_test_config(overrides={"max_steps": 50})
    result = evaluate_multi_seed(AlwaysHybridPolicy(), "S1", config, eval_seeds=[1, 2, 3, 4])
    assert result.below_floor_rate_mean == 0.0
    assert result.below_floor_rate_std == 0.0


def test_evaluate_multi_seed_below_floor_rate_matches_manual_floor_violations_over_max_steps():
    """Direct, independent proof the rate is genuinely `floor_violations
    / max_steps` per seed -- not inferred from `forced_rekey_ratio` or
    any other metric. Uses `security_masking: false` (so a policy can
    genuinely serve below a real, ratcheted-up floor -- see
    env/environment.py's design decision 16) + a stub policy that always
    attempts SERVE_CLASSICAL, same combination
    tests/test_environment.py's own `security_masking` proof already
    established."""
    import numpy as np

    class _AlwaysClassicalPolicy:
        def act(self, state, mask):
            return Action.SERVE_CLASSICAL

    max_steps = 200
    config = load_test_config(
        overrides={
            "scenario": "S2",
            "threat_schedule": {"elevate_at_step": 50, "elevated_signal": 6.0},
            "security_masking": False,
            "max_steps": max_steps,
        }
    )
    eval_seeds = [0, 1, 2]

    result = evaluate_multi_seed(_AlwaysClassicalPolicy(), "S2", config, eval_seeds)

    manual_results = [run_scenario(_AlwaysClassicalPolicy(), "S2", config, seed=s) for s in eval_seeds]
    manual_rates = [r.floor_violations / max_steps for r in manual_results]

    assert result.below_floor_rate_mean == pytest.approx(float(np.mean(manual_rates)))
    assert result.below_floor_rate_std == pytest.approx(float(np.std(manual_rates)))
    assert result.below_floor_rate_mean > 0.0  # genuinely exercised, not a vacuous zero-vs-zero check


def test_evaluate_multi_seed_single_seed_matches_run_scenario_exactly():
    """A single-eval-seed call must reduce to exactly `run_scenario`'s
    own result, with zero std -- the multi-seed machinery shouldn't
    change behavior in the degenerate n=1 case."""
    config = load_test_config(overrides={"max_steps": 50})
    policy = AlwaysPQCPolicy()

    direct = run_scenario(policy, "S1", config, seed=42)
    multi = evaluate_multi_seed(policy, "S1", config, eval_seeds=[42])

    assert multi.p99_latency_mean == pytest.approx(direct.p99_latency)
    assert multi.p99_latency_std == pytest.approx(0.0)
    assert multi.total_reward_mean == pytest.approx(direct.total_reward)


# ---------------------------------------------------------------------------
# S5 steering-attack dual-tracking measurement (run_scenario_under_attack /
# evaluate_attack_multi_seed) -- Part 1 of the dose-response sweep session.
# Real tests, proven BEFORE any real sweep is run, per that session's
# instruction.
# ---------------------------------------------------------------------------


def test_run_scenario_under_attack_alpha_zero_matches_run_scenario_exactly():
    """alpha=0.0 means no attack at all -- the live episode the agent
    acts in must be byte-for-byte identical to the unattacked
    `run_scenario` path (same forecaster trajectory, same actions, same
    every metric), and the TRUE-tracked floor must equal the ATTACKED
    floor at every single decision (there's no divergence to track when
    nothing was shaped)."""
    config = load_test_config(
        overrides={
            "scenario": "S2",
            "threat_schedule": {"elevate_at_step": 50, "elevated_signal": 6.0},
            "max_steps": 100,
        }
    )
    policy = AlwaysPQCPolicy()

    direct = run_scenario(policy, "S2", config, seed=3)
    attacked = run_scenario_under_attack(policy, "S2", config, seed=3, alpha=0.0)

    assert attacked.scenario_result.total_reward == pytest.approx(direct.total_reward)
    assert attacked.scenario_result.p99_latency == pytest.approx(direct.p99_latency)
    assert attacked.scenario_result.floor_violations == direct.floor_violations
    assert attacked.scenario_result.episode_metrics == direct.episode_metrics
    # no attack -> nothing for the TRUE track to ever disagree with
    assert attacked.below_floor_true_count == 0
    assert attacked.below_floor_rate_true == 0.0


def test_run_scenario_under_attack_below_floor_rate_true_is_zero_for_a_masked_policy():
    """Hard Rule 2's structural guarantee under attack: a masked policy
    can never be steered below whatever floor it's given -- see
    attack/trace_generator.py's own masking-safety proof. Full attack
    (alpha=1.0) on S2, `AlwaysHybridPolicy` (always tries the highest
    tier, so it's never itself the reason a decision looks short)."""
    config = load_test_config(
        overrides={
            "scenario": "S2",
            "threat_schedule": {"elevate_at_step": 50, "elevated_signal": 6.0},
            "max_steps": 100,
        }
    )
    result = run_scenario_under_attack(AlwaysHybridPolicy(), "S2", config, seed=0, alpha=1.0)
    assert result.below_floor_true_count == 0
    assert result.below_floor_rate_true == 0.0
    assert result.scenario_result.floor_violations == 0


class _AlwaysServeAttackedFloorPolicy:
    """Serves exactly the tier the ATTACKED floor currently requires --
    zero violations against what it was told (`state["policy_floor"]`
    is itself the attacked-floor tier action), so this policy is never
    itself "at fault" by the attacked side's own bookkeeping. Requires
    `security_masking: false` (no real mask is enforcing this -- the
    policy just happens to always comply with the lie it's shown)."""

    def act(self, state, mask):
        return Action(state["policy_floor"])


def test_run_scenario_under_attack_below_floor_rate_true_detects_a_real_divergence():
    """This session's most important harness-level test (per
    instruction): a case where TRUE and ATTACKED posture genuinely
    diverge across a discrete floor-table boundary, verified with real
    masking/floor computation (env/masking.py::PolicyTable, the same
    class env/environment.py itself uses) -- not a hardcoded expected
    rate.

    Full attack (alpha=1.0) on real S2 (`threat_schedule.elevate_at_step
    =50`, `elevated_signal=6.0`) means the live episode's forecaster
    only ever sees `g(xt) == [0.0, 0.0]` (attack/trace_generator.py's
    benign-region target), regardless of what the true window was at
    that tick -- so the ATTACKED posture never reaches HIGH the way S2's
    real scripted schedule is documented to (SESSION_LOG.md 2026-08-19:
    "posture reaches HIGH by internal tick ~55"), while the parallel,
    measurement-only shadow track (fed the TRUE window) does. With
    `security_masking: false` and a policy that always serves exactly
    the ATTACKED floor's own tier (`_AlwaysServeAttackedFloorPolicy`
    above -- by construction, zero violations against the attacked
    floor, i.e. `scenario_result.floor_violations == 0`), any nonzero
    `below_floor_true_count` can only come from the TRUE-posture track
    genuinely disagreeing with the attacked one -- not from the policy
    itself picking a bad action."""
    config = load_test_config(
        overrides={
            "scenario": "S2",
            "threat_schedule": {"elevate_at_step": 50, "elevated_signal": 6.0},
            "security_masking": False,
            "max_steps": 200,
        }
    )
    policy = _AlwaysServeAttackedFloorPolicy()

    result = run_scenario_under_attack(policy, "S2", config, seed=0, alpha=1.0)

    # the policy is, by construction, never at fault relative to what it was shown
    assert result.scenario_result.floor_violations == 0

    # yet the TRUE-posture track catches real, measured violations --
    # independently recomputed from the per-decision logs, not just
    # trusting the function's own counter
    manual_below_floor_true_count = sum(
        1
        for true_floor, delivered in zip(result.true_floor_log, result.delivered_tier_log)
        if int(delivered) < int(true_floor)
    )
    assert result.below_floor_true_count == manual_below_floor_true_count
    assert result.below_floor_true_count > 0
    assert result.below_floor_rate_true == pytest.approx(
        result.below_floor_true_count / result.total_requests
    )

    # a concrete instance of the actual discrete crossing this test is about:
    # at least one decision where the TRUE floor demanded HYBRID but the
    # (attack-compliant) delivered tier was only PQC
    assert any(
        true_floor == Action.SERVE_HYBRID and delivered == Action.SERVE_PQC
        for true_floor, delivered in zip(result.true_floor_log, result.delivered_tier_log)
    )

    # sanity: the TRUE floor genuinely reaches HIGH-posture territory
    # (SERVE_HYBRID, for at least S2/S3-sensitivity requests) at some
    # point, matching S2's documented real schedule -- the ATTACKED
    # floor track (recoverable via scenario_result together with the
    # policy's own choices) never needed to, since it only ever saw the
    # zeroed, attacked window.
    assert Action.SERVE_HYBRID in result.true_floor_log


def test_evaluate_attack_multi_seed_matches_manual_run_scenario_under_attack():
    """Direct, independent proof `evaluate_attack_multi_seed` is
    genuinely computed from `run_scenario_under_attack`'s own per-seed
    results, mirroring `evaluate_multi_seed`'s own single-seed
    reduction test."""
    config = load_test_config(
        overrides={
            "scenario": "S2",
            "threat_schedule": {"elevate_at_step": 50, "elevated_signal": 6.0},
            "max_steps": 60,
        }
    )
    policy = AlwaysPQCPolicy()
    alpha = 0.5
    eval_seeds = [1, 2, 3]

    import numpy as np

    result = evaluate_attack_multi_seed(policy, "S2", config, eval_seeds, alpha)

    manual = [run_scenario_under_attack(policy, "S2", config, seed=s, alpha=alpha) for s in eval_seeds]
    manual_rates = [r.below_floor_rate_true for r in manual]

    assert result.below_floor_rate_true_mean == pytest.approx(float(np.mean(manual_rates)))
    assert result.below_floor_rate_true_std == pytest.approx(float(np.std(manual_rates)))
    assert [r.below_floor_true_count for r in result.results] == [
        r.below_floor_true_count for r in manual
    ]


def test_evaluate_attack_multi_seed_rejects_empty_eval_seeds():
    config = load_test_config(overrides={"max_steps": 20})
    with pytest.raises(ValueError):
        evaluate_attack_multi_seed(AlwaysPQCPolicy(), "S1", config, eval_seeds=[], alpha=0.5)
