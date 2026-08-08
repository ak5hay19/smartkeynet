"""Behavioral tests for `agents.dqn` -- the masked DQN agent (PLAN.md §4
architecture diagram "DQN AGENT"; §10 kickoff step 5).

Proves the building blocks work (state flattening, masked action
selection at both epsilon extremes, replay buffer accumulation, a real
gradient step moving weights and Q-values in the right direction,
checkpoint round-tripping, and a short real-environment training loop
whose loss trends down) -- not a full training campaign to convergence.
That's `experiments/train.py`, deliberately a separate future session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from agents.dqn import (
    DQNAgent,
    DQNConfig,
    QNetwork,
    _FORECAST_STATE_DIM,
    _OFF_STATE_DIM,
    flatten_state,
    load_dqn_config,
)
from env.contracts import N_ACTIONS, Action
from env.environment import SmartKeyNetEnv


def _make_state(
    *,
    forecast: bool,
    pool_fill: float = 0.5,
    key_age: float = 10.0,
    key_type_onehot: tuple[float, float, float] = (0.0, 1.0, 0.0),
    sensitivity_class: int = 1,
    policy_floor: int = 1,
    regret_event_recent: bool = False,
) -> dict[str, Any]:
    """Minimal `StateDict` stand-in (runtime it's just a dict). Mirrors
    exactly what env/environment.py's `_prepare_decision` actually
    zeroes under `off` vs. populates under `ewma` -- see
    tests/test_environment.py's `test_foresight_fields_zeroed_under_off`
    / `test_foresight_fields_populated_under_ewma`."""
    if forecast:
        threat_score = 0.42
        threat_forecast = [0.1, 0.2, 0.3, 0.4, 0.5]
        pool_level_hat = [0.3, 0.4, 0.5]
        skr_mean_hat = [1.0, 1.1, 1.2]
        hybrid_demand_hat = [2.0, 3.0, 4.0]
    else:
        threat_score = 0.0
        threat_forecast = [0.0] * 5
        pool_level_hat = [0.0] * 3
        skr_mean_hat = [0.0] * 3
        hybrid_demand_hat = [0.0] * 3

    return {
        "threat_score": threat_score,
        "threat_forecast": threat_forecast,
        "qber": 0.01,
        "skr": 500.0,
        "pool_fill": pool_fill,
        "arrival_rate": 1.0,
        "load": 0.2,
        "avg_latency": 1.0,
        "key_age": key_age,
        "key_type_onehot": list(key_type_onehot),
        "sensitivity_class": sensitivity_class,
        "policy_floor": policy_floor,
        "pool_level_hat": pool_level_hat,
        "skr_mean_hat": skr_mean_hat,
        "hybrid_demand_hat": hybrid_demand_hat,
        "regret_event_recent": regret_event_recent,
    }


def _full_mask() -> np.ndarray:
    return np.ones(N_ACTIONS, dtype=bool)


def _mask(*legal_actions: Action) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for action in legal_actions:
        mask[int(action)] = True
    return mask


# ---------------------------------------------------------------------------
# flatten_state
# ---------------------------------------------------------------------------


def test_flatten_state_off_mode_excludes_forecast_fields():
    state = _make_state(forecast=False)
    tensor = flatten_state(state)
    assert tensor.shape == (_OFF_STATE_DIM,)
    assert tensor.shape == (13,)


def test_flatten_state_forecast_mode_includes_forecast_fields():
    state = _make_state(forecast=True)
    tensor = flatten_state(state)
    assert tensor.shape == (_FORECAST_STATE_DIM,)
    assert tensor.shape == (28,)


def test_flatten_state_forecast_values_actually_present_not_just_longer():
    """Not just a longer vector -- the forecast values themselves must
    actually appear in it (a real dimensionality difference, not
    cosmetic zero-padding)."""
    state = _make_state(forecast=True)
    tensor = flatten_state(state)
    values = tensor.tolist()
    assert any(abs(v - 0.42) < 1e-6 for v in values)  # threat_score
    assert any(abs(v - 4.0) < 1e-6 for v in values)  # hybrid_demand_hat[-1]


# ---------------------------------------------------------------------------
# QNetwork
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state_dim", [_OFF_STATE_DIM, _FORECAST_STATE_DIM])
def test_qnetwork_forward_shape(state_dim):
    net = QNetwork(state_dim)
    x = torch.zeros(1, state_dim)
    out = net(x)
    assert out.shape == (1, N_ACTIONS)


def test_qnetwork_forward_batched():
    net = QNetwork(_OFF_STATE_DIM)
    x = torch.zeros(7, _OFF_STATE_DIM)
    out = net(x)
    assert out.shape == (7, N_ACTIONS)


# ---------------------------------------------------------------------------
# DQNConfig / load_dqn_config
# ---------------------------------------------------------------------------


def test_load_dqn_config_matches_real_yaml():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)["dqn"]

    config = load_dqn_config()
    assert config.gamma == raw["gamma"]
    assert config.lr == raw["lr"]
    assert config.batch_size == raw["batch_size"]
    assert config.target_update_every == raw["target_update_every"]
    assert config.epsilon_start == raw["epsilon_start"]
    assert config.epsilon_end == raw["epsilon_end"]
    assert config.epsilon_decay_steps == raw["epsilon_decay_steps"]


# ---------------------------------------------------------------------------
# act() -- epsilon extremes, adversarial masks
# ---------------------------------------------------------------------------


def _greedy_config() -> DQNConfig:
    return DQNConfig(epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_steps=1, batch_size=4)


def _random_config() -> DQNConfig:
    return DQNConfig(epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_steps=1, batch_size=4)


def test_act_epsilon_zero_picks_the_highest_legal_q_value_full_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=_greedy_config())
    state = _make_state(forecast=False)

    with torch.no_grad():
        q_values = agent.q_network(flatten_state(state).unsqueeze(0)).squeeze(0)
    expected = int(torch.argmax(q_values).item())

    action = agent.act(state, _full_mask())
    assert int(action) == expected


def test_act_epsilon_zero_respects_a_restrictive_mask():
    """Excluding the network's globally-preferred action from the mask
    must make act() fall back to the best *legal* one, not the global
    argmax."""
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=_greedy_config())
    state = _make_state(forecast=False)

    with torch.no_grad():
        q_values = agent.q_network(flatten_state(state).unsqueeze(0)).squeeze(0)
    global_best = int(torch.argmax(q_values).item())

    restrictive_mask = _full_mask()
    restrictive_mask[global_best] = False
    assert restrictive_mask.any()

    remaining_q = q_values.clone()
    remaining_q[~torch.as_tensor(restrictive_mask)] = float("-inf")
    expected_second_best = int(torch.argmax(remaining_q).item())

    action = agent.act(state, restrictive_mask)
    assert int(action) == expected_second_best
    assert restrictive_mask[int(action)]


def test_act_epsilon_zero_single_legal_action_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=_greedy_config())
    state = _make_state(forecast=False)
    mask = _mask(Action.REKEY_NOW)
    assert agent.act(state, mask) == Action.REKEY_NOW


def test_act_epsilon_one_always_returns_a_legal_action_full_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=_random_config())
    state = _make_state(forecast=False)
    mask = _full_mask()
    seen = set()
    for _ in range(200):
        action = agent.act(state, mask)
        assert mask[int(action)]
        seen.add(action)
    assert seen == set(Action)  # every legal option actually gets drawn eventually


def test_act_epsilon_one_respects_a_restrictive_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=_random_config())
    state = _make_state(forecast=False)
    mask = _mask(Action.SERVE_PQC, Action.REKEY_NOW)
    seen = set()
    for _ in range(200):
        action = agent.act(state, mask)
        assert mask[int(action)]
        seen.add(action)
    assert seen == {Action.SERVE_PQC, Action.REKEY_NOW}


def test_act_epsilon_one_single_legal_action_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=_random_config())
    state = _make_state(forecast=False)
    mask = _mask(Action.REUSE)
    for _ in range(20):
        assert agent.act(state, mask) == Action.REUSE


def test_act_raises_on_all_illegal_mask():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=_greedy_config())
    state = _make_state(forecast=False)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    with pytest.raises(ValueError):
        agent.act(state, mask)


# ---------------------------------------------------------------------------
# observe()
# ---------------------------------------------------------------------------


def test_observe_accumulates_transitions_in_replay_buffer():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=DQNConfig(batch_size=4))
    state = _make_state(forecast=False)
    next_state = _make_state(forecast=False)
    mask = _full_mask()

    assert len(agent._replay_buffer) == 0
    for _ in range(5):
        agent.observe(state, Action.SERVE_PQC, 1.0, next_state, mask, False)
    assert len(agent._replay_buffer) == 5


# ---------------------------------------------------------------------------
# learn()
# ---------------------------------------------------------------------------


def test_learn_is_a_noop_below_batch_size():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=DQNConfig(batch_size=64))
    metrics = agent.learn()
    assert metrics["loss"] == 0.0


def test_learn_changes_network_weights():
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=DQNConfig(batch_size=4))
    state = _make_state(forecast=False)
    next_state = _make_state(forecast=False)
    mask = _full_mask()

    for _ in range(10):
        agent.observe(state, Action.SERVE_PQC, 1.0, next_state, mask, False)

    before = [p.detach().clone() for p in agent.q_network.parameters()]
    metrics = agent.learn()
    after = list(agent.q_network.parameters())

    assert "loss" in metrics
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_learn_moves_q_values_toward_the_obviously_correct_answer():
    """Not full convergence -- proves the loop works: repeatedly told
    "action A -> +10, action B -> -10" for the same state, the network
    should end up ranking A above B."""
    torch.manual_seed(0)
    config = DQNConfig(batch_size=8, gamma=0.0, lr=0.02, target_update_every=10_000)
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=config)

    state = _make_state(forecast=False)
    next_state = _make_state(forecast=False)  # irrelevant: gamma=0, done=True zeroes the bootstrap term
    mask = _full_mask()

    good_action = Action.SERVE_PQC
    bad_action = Action.SERVE_CLASSICAL

    for _ in range(400):
        agent.observe(state, good_action, 10.0, next_state, mask, True)
        agent.observe(state, bad_action, -10.0, next_state, mask, True)

    for _ in range(400):
        agent.learn()

    with torch.no_grad():
        q = agent.q_network(flatten_state(state).unsqueeze(0)).squeeze(0)

    assert q[int(good_action)] > q[int(bad_action)]


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------


def test_save_then_load_produces_identical_q_values(tmp_path):
    agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=DQNConfig(batch_size=4))
    state = _make_state(forecast=False)
    next_state = _make_state(forecast=False)
    mask = _full_mask()

    for _ in range(10):
        agent.observe(state, Action.SERVE_PQC, 1.0, next_state, mask, False)
    agent.learn()  # give it non-default (trained) weights, not just fresh init

    checkpoint_path = tmp_path / "agent.pt"
    agent.save(str(checkpoint_path))

    fresh_agent = DQNAgent(state_dim=_OFF_STATE_DIM, config=DQNConfig(batch_size=4))
    fresh_agent.load(str(checkpoint_path))

    probe = flatten_state(_make_state(forecast=False, pool_fill=0.9)).unsqueeze(0)
    with torch.no_grad():
        q1 = agent.q_network(probe)
        q2 = fresh_agent.q_network(probe)

    assert torch.allclose(q1, q2)


# ---------------------------------------------------------------------------
# Integration: real SmartKeyNetEnv, S1, loss trends down
# ---------------------------------------------------------------------------


def test_dqn_agent_loss_trends_down_training_against_real_env_s1():
    """PLAN.md §10 step 5's "prove the loop works" evidence: not a full
    training campaign to convergence (that's experiments/train.py, a
    separate future session) -- just observe()+learn() every step for
    a few thousand real S1 steps and confirm the loss trends down
    rather than staying flat or diverging.

    Both the env and the agent's network init are seeded, so this is a
    deterministic run, not a flaky stochastic one -- if it passes once
    it passes identically every time.
    """
    torch.manual_seed(0)

    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    dqn_config = load_dqn_config()
    dqn_config.epsilon_decay_steps = 1000  # short run -- reach low epsilon within budget, don't hardcode everything else

    env = SmartKeyNetEnv({**full_config, "seed": 0, "max_steps": 3000})
    state, info = env.reset(seed=0)
    mask = info["action_mask"]

    state_dim = flatten_state(state).shape[0]  # derived from the real state, not assumed
    agent = DQNAgent(state_dim=state_dim, config=dqn_config)

    losses: list[float] = []
    truncated = False
    while not truncated:
        action = agent.act(state, mask)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_mask = info["action_mask"]

        agent.observe(state, action, reward, next_state, next_mask, terminated)
        metrics = agent.learn()
        if metrics["loss"] > 0.0:
            losses.append(metrics["loss"])

        state = next_state
        mask = next_mask

    assert len(losses) > 500  # learning actually ran for a meaningful stretch, not just a handful of steps

    # skip an early warmup window (network still close to random init,
    # epsilon still high) before comparing early vs. late loss -- a
    # generous, directional check, not a tight convergence bound.
    warmup = len(losses) // 10
    window = len(losses) // 4
    early = losses[warmup : warmup + window]
    late = losses[-window:]

    early_avg = sum(early) / len(early)
    late_avg = sum(late) / len(late)
    assert late_avg < early_avg, f"loss did not trend down: early={early_avg:.4f}, late={late_avg:.4f}"
