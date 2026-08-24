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
