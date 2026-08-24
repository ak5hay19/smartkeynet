"""Behavioral tests for `experiments.train` -- the S1 overfit-on-purpose
training campaign (PLAN.md §10 step 5).

Kept CI-fast on purpose: these are smoke tests over a handful of steps,
not a training campaign to convergence -- that's a separate, manual,
much-longer run (see SESSION_LOG.md for the actually-achieved numbers
from running `python -m experiments.train` for real).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from agents.dqn import DQNAgent, DQNConfig
from env.contracts import N_ACTIONS
from experiments.harness import ScenarioResult
from experiments.train import GreedyDQNPolicy, load_full_config, train


def _make_state(*, pool_fill: float = 0.5) -> dict[str, Any]:
    """Minimal off-mode (`has_forecast=False`, 13-dim) `StateDict`
    stand-in -- runtime it's just a dict. Mirrors tests/test_dqn.py's
    own `_make_state` helper, trimmed to only what this file needs."""
    return {
        "threat_score": 0.0,
        "threat_forecast": [0.0] * 5,
        "qber": 0.01,
        "skr": 500.0,
        "pool_fill": pool_fill,
        "arrival_rate": 1.0,
        "load": 0.2,
        "avg_latency": 1.0,
        "key_age": 10.0,
        "key_type_onehot": [0.0, 1.0, 0.0],
        "sensitivity_class": 1,
        "policy_floor": 1,
        "pool_level_hat": [0.0] * 3,
        "skr_mean_hat": [0.0] * 3,
        "hybrid_demand_hat": [0.0] * 3,
        "regret_event_recent": False,
    }


def _full_mask() -> np.ndarray:
    return np.ones(N_ACTIONS, dtype=bool)


# ---------------------------------------------------------------------------
# train() smoke test
# ---------------------------------------------------------------------------


def test_train_smoke_run_produces_checkpoint_and_tracks_metrics(tmp_path):
    """A short run (well past DQNConfig's default batch_size=64, so
    `learn()` actually takes real gradient steps, not just no-ops)
    completes without crashing, saves a checkpoint, and leaves behind
    real tracked loss/reward/eval-snapshot data -- not empty lists."""
    full_config = load_full_config()
    checkpoint_path = tmp_path / "dqn_smoke.pt"

    overrides = {
        "total_steps": 100,
        "eval_every": 50,
        "eval_max_steps": 20,
        "checkpoint_path": str(checkpoint_path),
    }

    agent, record = train(full_config, training_overrides=overrides)

    assert isinstance(agent, DQNAgent)
    assert checkpoint_path.exists()
    assert record.checkpoint_path == str(checkpoint_path)

    # learning actually happened: buffer fills at batch_size=64 (the
    # real configs/default.yaml dqn.batch_size), so steps 65-100 take
    # real gradient steps.
    assert len(record.losses) > 0
    assert len(record.loss_steps) == len(record.losses)

    # reward is tracked every eval_every window, not just at the end
    assert len(record.reward_window_avgs) == 2  # steps 50 and 100

    # periodic greedy-mode eval snapshots ran via the real harness
    assert len(record.eval_snapshots) == 2
    seen_steps = [step for step, _ in record.eval_snapshots]
    assert seen_steps == [50, 100]
    for _step, result in record.eval_snapshots:
        assert isinstance(result, ScenarioResult)
        assert result.scenario == "S1"
        assert result.floor_violations == 0  # masking architecture holds for a trained agent too


def test_train_respects_explicit_total_steps_not_the_real_default():
    """`training_overrides["total_steps"]` must actually shrink the run
    -- not silently fall back to configs/default.yaml's real (much
    longer) default."""
    full_config = load_full_config()
    overrides = {
        "total_steps": 70,
        "eval_every": 70,
        "eval_max_steps": 10,
        "checkpoint_path": "unused-in-this-test.pt",
    }
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        overrides["checkpoint_path"] = str(Path(tmp_dir) / "agent.pt")
        _agent, record = train(full_config, training_overrides=overrides)

    assert len(record.eval_snapshots) == 1
    assert record.eval_snapshots[0][0] == 70


# ---------------------------------------------------------------------------
# GreedyDQNPolicy -- deterministic eval vs. stochastic training act()
# ---------------------------------------------------------------------------


def test_greedy_policy_is_deterministic_unlike_stochastic_training_act():
    """The resolution to point 3's design question: `GreedyDQNPolicy`
    must return the same action every time for a fixed state/mask,
    while `DQNAgent.act()` itself (epsilon=1 here, to make the
    contrast unambiguous) is genuinely stochastic across repeated
    calls with that same state/mask."""
    stochastic_config = DQNConfig(epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_steps=1, batch_size=4)
    agent = DQNAgent(state_dim=13, has_forecast=False, config=stochastic_config)

    state = _make_state()
    mask = _full_mask()

    greedy_policy = GreedyDQNPolicy(agent)
    greedy_actions = {greedy_policy.act(state, mask) for _ in range(30)}
    assert len(greedy_actions) == 1  # deterministic: same state -> same action, every time

    stochastic_actions = {agent.act(state, mask) for _ in range(50)}
    assert len(stochastic_actions) > 1  # genuinely stochastic under epsilon=1, unlike the greedy wrapper


# ---------------------------------------------------------------------------
# Hard Rule 8 guard (2026-08-24): "train on stationary scenarios; the
# migration-wave scenario is held-out evaluation only." -- see
# experiments/train.py::train()'s own comment for the full reasoning.
# ---------------------------------------------------------------------------


def test_hard_rule_8_train_refuses_the_real_committed_s6_config():
    """The single most important test this session produces (per
    instruction): attempting to train on the real, committed
    `configs/scenarios/s6_migration.yaml` must fail loudly and
    explicitly, not silently proceed. Must raise before any real
    training work happens -- this test would otherwise be slow if it
    didn't."""
    s6_config = load_full_config("configs/scenarios/s6_migration.yaml")
    assert s6_config["train_eligible"] is False  # the guard's own precondition, not assumed

    with pytest.raises(ValueError, match="train_eligible"):
        train(s6_config, training_overrides={"total_steps": 100, "eval_every": 50, "eval_max_steps": 10})


def test_hard_rule_8_guard_is_keyed_on_the_flag_not_a_hardcoded_scenario_string():
    """The guard must fire off the `train_eligible` flag itself, not a
    string match on `scenario == "S6"` -- so it stays correct even if a
    config sets `train_eligible: false` for some other reason/scenario
    in the future. A synthetic config makes this explicit."""
    full_config = load_full_config()
    full_config = {**full_config, "scenario": "S1", "train_eligible": False}

    with pytest.raises(ValueError, match="train_eligible"):
        train(full_config, training_overrides={"total_steps": 100, "eval_every": 50, "eval_max_steps": 10})


def test_hard_rule_8_guard_is_a_no_op_for_every_pre_existing_config():
    """Every config that doesn't set `train_eligible` (every one before
    this session) must be completely unaffected -- `.get(..., True)`
    defaults to eligible, not a new opt-in requirement."""
    for path in (
        "configs/default.yaml",
        "configs/scenarios/s2_hndl.yaml",
        "configs/scenarios/s3_degradation.yaml",
        "configs/scenarios/s4_ddos.yaml",
    ):
        config = load_full_config(path)
        assert config.get("train_eligible", True) is True


# ---------------------------------------------------------------------------
# `scenario` parameter (2026-08-24): a real, explicit parameter following
# experiments/harness.py's run_scenario/run_grid convention -- see
# experiments/train.py::train()'s own docstring. Defaults to "S1" so every
# pre-existing call site above (which never passes it) is unaffected --
# proven directly, not just asserted, by the byte-for-byte comparison test
# below.
# ---------------------------------------------------------------------------


def test_train_default_scenario_is_byte_for_byte_identical_to_pre_fix_s1_behavior():
    """Before this session, `train()` hardcoded `scenario: "S1"` --
    calling it without a `scenario` argument must still reproduce
    exactly the same trajectory (loss curve, reward-window averages,
    every eval snapshot's metrics) now that `scenario` is a real,
    external parameter defaulting to `"S1"`. This is a genuine
    before/after comparison (fixed seed makes both env and DQNAgent
    RNG fully reproducible -- see agents/dqn.py's seed docstring), not
    just 'tests still pass'."""
    full_config = load_full_config()
    overrides = {
        "total_steps": 300,
        "eval_every": 100,
        "eval_max_steps": 50,
        "checkpoint_path": "unused-in-this-test.pt",
    }
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        overrides["checkpoint_path"] = str(Path(tmp_dir) / "agent.pt")
        _agent, record = train(full_config, training_overrides=overrides)

    # Pinned exact values from a real pre-fix run (git-stashed this
    # session's diff and re-ran the identical config/seed/overrides
    # before writing this assertion -- not guessed).
    assert record.reward_window_avgs == [
        -140.727734,
        -115.35223400000001,
        -117.97983400000001,
    ]
    assert len(record.losses) == 237
    assert record.losses[0] == pytest.approx(36966.12890625)
    assert record.losses[-1] == pytest.approx(5178.3486328125)

    seen_steps = [step for step, _ in record.eval_snapshots]
    assert seen_steps == [100, 200, 300]
    expected = [
        (-2645.3858, 1.5, 0.5428571428571428),
        (-2629.9248, 1.5, 0.6785714285714286),
        (-2645.3858, 1.5, 0.5428571428571428),
    ]
    for (_step, result), (exp_reward, exp_p99, exp_fr) in zip(record.eval_snapshots, expected):
        assert result.scenario == "S1"
        assert result.total_reward == pytest.approx(exp_reward)
        assert result.p99_latency == pytest.approx(exp_p99)
        assert result.episode_metrics.forced_rekey_ratio == pytest.approx(exp_fr)


def test_train_scenario_s3_dispatches_real_s3_degradation_dynamics(tmp_path):
    """Passing `scenario="S3"` must actually reach `SmartKeyNetEnv`'s
    real S3 dispatch (env/environment.py), not just be accepted without
    error -- proven two ways: (1) S3 dispatch requires `qkd_degradation`
    in config, which `configs/default.yaml` doesn't have, so pointing
    `scenario="S3"` at the S1 default config must raise `KeyError`
    (proof the parameter is load-bearing, not ignored); (2) training
    against the real `configs/scenarios/s3_degradation.yaml` with
    `scenario="S3"` produces eval snapshots whose own `ScenarioResult.
    scenario` field reports `"S3"`, confirming the periodic eval
    snapshots (not just the training env) follow the same parameter."""
    default_config = load_full_config()
    s3_config = load_full_config("configs/scenarios/s3_degradation.yaml")

    with pytest.raises(KeyError, match="qkd_degradation"):
        train(
            default_config,
            training_overrides={
                "total_steps": 5,
                "eval_every": 5,
                "eval_max_steps": 5,
                "checkpoint_path": str(tmp_path / "unused.pt"),
            },
            scenario="S3",
        )

    _agent, record = train(
        s3_config,
        training_overrides={
            "total_steps": 80,
            "eval_every": 80,
            "eval_max_steps": 30,
            "checkpoint_path": str(tmp_path / "s3.pt"),
        },
        scenario="S3",
    )
    assert len(record.eval_snapshots) == 1
    _step, result = record.eval_snapshots[0]
    assert result.scenario == "S3"


def test_train_scenario_s3_pool_trajectory_genuinely_diverges_from_s1():
    """Direct mechanism-level proof (not just 'the field says S3'):
    constructing the real env the same way `train()` does internally
    (`{**full_config, "scenario": scenario, "seed": seed}`) and running
    a deterministic max-hybrid-draw stress test (mirrors SESSION_LOG.md
    2026-08-24's own S3-vs-S1 divergence methodology) shows S3's pool
    collapsing to near-total exhaustion while S1's stays comfortably
    full, under the real committed config files."""
    from env.contracts import Action
    from env.environment import SmartKeyNetEnv

    default_config = load_full_config()
    s3_config = load_full_config("configs/scenarios/s3_degradation.yaml")

    def min_pool_fill(full_config: dict[str, Any], scenario: str, seed: int) -> float:
        env_config = {**full_config, "scenario": scenario, "seed": seed}
        env = SmartKeyNetEnv(env_config)
        state, info = env.reset(seed=seed)
        mask = info["action_mask"]
        min_fill = state["pool_fill"]
        for _ in range(250):
            action = (
                Action.SERVE_HYBRID
                if mask[int(Action.SERVE_HYBRID)]
                else next(Action(a) for a in range(len(mask)) if mask[a])
            )
            state, _reward, _terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            min_fill = min(min_fill, state["pool_fill"])
            if truncated:
                break
        return min_fill

    s1_min_fill = min_pool_fill(default_config, "S1", seed=0)
    s3_min_fill = min_pool_fill(s3_config, "S3", seed=0)

    assert s1_min_fill > 0.5  # S1's pool never meaningfully stressed (pre-existing property)
    assert s3_min_fill < 0.01  # S3 collapses to near-total exhaustion (2026-08-24 recalibration)
    assert s3_min_fill < s1_min_fill


def test_train_scenario_s6_guard_fires_before_any_env_or_training_work(monkeypatch, tmp_path):
    """The single most important test this session produces: calling
    `train()` with `scenario="S6"` selected explicitly (not just an S6
    config happening to be passed) must raise `ValueError` before a
    real `SmartKeyNetEnv` is ever constructed -- proving the
    `train_eligible` guard blocks a genuinely scenario-selectable call,
    not just a code path nobody could previously reach (see
    SESSION_LOG.md 2026-08-24's S6 session, which built the guard but
    could never exercise it through a real, general-purpose `train()`
    call, since `scenario` was hardcoded to `"S1"` at the time)."""
    from env.environment import SmartKeyNetEnv

    call_count = {"init": 0}
    orig_init = SmartKeyNetEnv.__init__

    def counting_init(self, *args, **kwargs):
        call_count["init"] += 1
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(SmartKeyNetEnv, "__init__", counting_init)

    s6_config = load_full_config("configs/scenarios/s6_migration.yaml")
    with pytest.raises(ValueError, match="train_eligible"):
        train(
            s6_config,
            training_overrides={
                "total_steps": 100,
                "eval_every": 50,
                "eval_max_steps": 10,
                "checkpoint_path": str(tmp_path / "s6.pt"),
            },
            scenario="S6",
        )

    assert call_count["init"] == 0  # no environment ever constructed -- no env steps, no optimizer steps


def test_train_scenario_s3_control_does_not_raise_guard_is_scenario_specific(tmp_path):
    """Control test: `scenario="S3"` (`train_eligible` defaults `True`
    there) must NOT raise -- proving the Hard Rule 8 guard is
    scenario-specific (keyed on the config's own `train_eligible` flag,
    per `configs/scenarios/s6_migration.yaml`), not a blanket block on
    every non-S1 scenario."""
    s3_config = load_full_config("configs/scenarios/s3_degradation.yaml")
    agent, record = train(
        s3_config,
        training_overrides={
            "total_steps": 80,
            "eval_every": 80,
            "eval_max_steps": 10,
            "checkpoint_path": str(tmp_path / "s3_control.pt"),
        },
        scenario="S3",
    )
    assert isinstance(agent, DQNAgent)
    assert len(record.eval_snapshots) == 1


def test_greedy_policy_never_mutates_agent_act_call_counter():
    """`GreedyDQNPolicy` must not burn through the agent's training
    epsilon-decay budget -- it never calls `agent.act()`, so
    `agent._act_calls` (which drives epsilon decay) must stay
    untouched by eval snapshots."""
    agent = DQNAgent(state_dim=13, has_forecast=False, config=DQNConfig(batch_size=4))
    state = _make_state()
    mask = _full_mask()

    assert agent._act_calls == 0
    greedy_policy = GreedyDQNPolicy(agent)
    for _ in range(10):
        greedy_policy.act(state, mask)
    assert agent._act_calls == 0


# ---------------------------------------------------------------------------
# train_soft_reward_baseline (2026-08-25) -- agents/soft_reward_baseline.py's
# training entry point. Additive: none of the tests above this point are
# touched or affected.
# ---------------------------------------------------------------------------


def test_train_soft_reward_baseline_smoke_run_produces_checkpoint_and_tracks_metrics(tmp_path):
    """Mirrors test_train_smoke_run_produces_checkpoint_and_tracks_metrics
    above, for the soft-reward baseline's own training entry point."""
    from experiments.train import train_soft_reward_baseline

    full_config = load_full_config("configs/soft_reward_baseline.yaml")
    checkpoint_path = tmp_path / "soft_reward_smoke.pt"

    overrides = {
        "total_steps": 100,
        "eval_every": 50,
        "eval_max_steps": 20,
        "checkpoint_path": str(checkpoint_path),
    }

    agent, record = train_soft_reward_baseline(full_config, training_overrides=overrides)

    assert isinstance(agent, DQNAgent)
    assert checkpoint_path.exists()
    assert record.checkpoint_path == str(checkpoint_path)
    assert len(record.losses) > 0
    assert len(record.reward_window_avgs) == 2
    assert len(record.eval_snapshots) == 2


def test_train_soft_reward_baseline_uses_security_masking_false_environment(monkeypatch, tmp_path):
    """Direct dispatch proof: the environment this function constructs
    must genuinely have `security_masking: False` wired through, not
    silently ignored -- checked via the real `SmartKeyNetEnv.__init__`,
    not assumed from the config file's own contents."""
    from env.environment import SmartKeyNetEnv
    from experiments.train import train_soft_reward_baseline

    seen_configs: list[dict[str, Any]] = []
    orig_init = SmartKeyNetEnv.__init__

    def capturing_init(self, config, *args, **kwargs):
        seen_configs.append(config)
        return orig_init(self, config, *args, **kwargs)

    monkeypatch.setattr(SmartKeyNetEnv, "__init__", capturing_init)

    full_config = load_full_config("configs/soft_reward_baseline.yaml")
    train_soft_reward_baseline(
        full_config,
        training_overrides={
            "total_steps": 30,
            "eval_every": 30,
            "eval_max_steps": 10,
            "checkpoint_path": str(tmp_path / "masking_check.pt"),
        },
    )

    # Two SmartKeyNetEnv constructions expected: the training env, plus
    # one per eval snapshot (run_scenario constructs its own).
    assert len(seen_configs) >= 2
    assert all(cfg["security_masking"] is False for cfg in seen_configs)


def test_train_soft_reward_baseline_does_not_use_env_steps_own_reward(monkeypatch, tmp_path):
    """Hard Rule 1 boundary proof at the training-loop level: whatever
    `env.step()` itself returns as `reward` must never reach
    `agent.observe()` for this agent -- only `compute_soft_reward`'s
    independently-computed value does. Verified by making `env.step()`
    return an impossible, obviously-wrong reward and confirming the
    replay buffer's stored transitions never contain it."""
    from env.environment import SmartKeyNetEnv
    from experiments.train import train_soft_reward_baseline

    poison_value = -999_999.0
    orig_step = SmartKeyNetEnv.step

    def poisoning_step(self, action):
        state, _reward, terminated, truncated, info = orig_step(self, action)
        return state, poison_value, terminated, truncated, info

    monkeypatch.setattr(SmartKeyNetEnv, "step", poisoning_step)

    full_config = load_full_config("configs/soft_reward_baseline.yaml")
    agent, _record = train_soft_reward_baseline(
        full_config,
        training_overrides={
            "total_steps": 80,
            "eval_every": 80,
            "eval_max_steps": 10,
            "checkpoint_path": str(tmp_path / "poison_check.pt"),
        },
    )

    stored_rewards = [t.reward for t in agent._replay_buffer._storage]
    assert len(stored_rewards) > 0
    assert poison_value not in stored_rewards


def test_train_soft_reward_baseline_trained_agent_serves_below_the_real_floor(tmp_path):
    """The actual proof this whole agent exists to produce: a real, short
    training run's periodic greedy-mode eval snapshots (via the
    unmodified `experiments/harness.py::run_scenario`, same as `train()`
    uses) must show `ScenarioResult.floor_violations > 0` for at least
    one snapshot -- direct, measured evidence this agent's security term
    is genuinely soft and gets traded away, not merely inferred from the
    reward formula's shape. Also confirms (a) loss trends in a sane
    direction (ends lower than it started, on average across the run's
    two halves) -- both real training-run guarantees this session's
    brief asked for, from one run."""
    from experiments.train import train_soft_reward_baseline

    full_config = load_full_config("configs/soft_reward_baseline.yaml")
    overrides = {
        "total_steps": 3000,
        "eval_every": 500,
        "eval_max_steps": 100,
        "checkpoint_path": str(tmp_path / "soft_reward_real_run.pt"),
    }

    agent, record = train_soft_reward_baseline(full_config, training_overrides=overrides)

    assert isinstance(agent, DQNAgent)
    assert len(record.losses) > 0

    any_below_floor = any(result.floor_violations > 0 for _step, result in record.eval_snapshots)
    assert any_below_floor, (
        "expected at least one eval snapshot with floor_violations > 0 -- "
        "direct evidence the soft-reward agent trades security away, which "
        "is the entire property this agent exists to demonstrate"
    )

    half = len(record.losses) // 2
    first_half_mean = sum(record.losses[:half]) / half
    second_half_mean = sum(record.losses[half:]) / (len(record.losses) - half)
    assert second_half_mean < first_half_mean, (
        f"loss did not trend down: first half mean={first_half_mean:.2f}, "
        f"second half mean={second_half_mean:.2f}"
    )


def test_train_soft_reward_baseline_respects_train_eligible_guard(tmp_path):
    """Mirrors train()'s own Hard Rule 8 guard test -- a soft-reward
    config with train_eligible: false must still refuse to train,
    proving the guard is shared behavior (both functions check the same
    full_config.get("train_eligible", True) convention), not something
    only train() enforces."""
    from experiments.train import train_soft_reward_baseline

    full_config = load_full_config("configs/soft_reward_baseline.yaml")
    full_config = {**full_config, "train_eligible": False}

    with pytest.raises(ValueError, match="train_eligible"):
        train_soft_reward_baseline(
            full_config,
            training_overrides={
                "total_steps": 10,
                "eval_every": 10,
                "eval_max_steps": 5,
                "checkpoint_path": str(tmp_path / "unused.pt"),
            },
        )


# ---------------------------------------------------------------------------
# configs/soft_reward_baseline_s3.yaml (2026-08-25, masked-vs-soft-reward
# S3 comparison session) -- the standalone S3 variant of
# configs/soft_reward_baseline.yaml, PROGRESS.md's "Next task" follow-up
# (1). Same standalone-config-per-scenario convention as
# configs/scenarios/s2_hndl.yaml etc.
# ---------------------------------------------------------------------------


def test_soft_reward_baseline_s3_config_loads_and_constructs_a_working_env():
    """The committed, standalone S3 config must actually be loadable and
    runnable, not just referenced -- mirrors
    tests/test_environment.py::test_scenario_config_files_load_and_construct_a_working_env
    for this file, which lives in configs/ rather than configs/scenarios/
    (matching configs/soft_reward_baseline.yaml's own location, not the
    per-scenario subfolder)."""
    from env.environment import SmartKeyNetEnv

    config = load_full_config("configs/soft_reward_baseline_s3.yaml")
    assert config["scenario"] == "S3"
    assert config["security_masking"] is False
    config["max_steps"] = 20

    env = SmartKeyNetEnv(config)
    assert env._scenario == "S3"
    assert env._security_masking is False
    state, info = env.reset(seed=0)
    assert info["action_mask"].any()


def test_train_soft_reward_baseline_s3_config_smoke_run(tmp_path):
    """A short real training run against the committed S3 config must
    complete without crashing (the concrete regression risk: S3's
    smaller, recalibrated pool interacting with `security_masking: false`
    -- e.g. a pool-exhaustion/deferral path colliding with an
    always-legal-SERVE_CLASSICAL mask -- see env/environment.py's design
    decision 16 for why pool/key-age feasibility rules stay enforced
    regardless of this flag)."""
    from experiments.train import train_soft_reward_baseline

    full_config = load_full_config("configs/soft_reward_baseline_s3.yaml")
    checkpoint_path = tmp_path / "soft_reward_s3_smoke.pt"

    agent, record = train_soft_reward_baseline(
        full_config,
        training_overrides={
            "total_steps": 100,
            "eval_every": 50,
            "eval_max_steps": 20,
            "checkpoint_path": str(checkpoint_path),
        },
        scenario="S3",
    )

    assert isinstance(agent, DQNAgent)
    assert checkpoint_path.exists()
    assert len(record.losses) > 0
    assert all(result.scenario == "S3" for _step, result in record.eval_snapshots)
